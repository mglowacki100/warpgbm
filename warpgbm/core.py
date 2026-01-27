import torch
import numpy as np
import pickle
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from warpgbm.cuda import node_kernel
from warpgbm.metrics import rmsle_torch, softmax, log_loss_torch, accuracy_torch
from tqdm import tqdm
from typing import Tuple
from torch import Tensor
import gc

class WarpGBM(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        objective="regression",
        num_bins=10,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=20,
        min_split_gain=0.0,
        threads_per_block=64,
        rows_per_thread=4,
        L2_reg=1e-6,
        device="cuda",
        colsample_bytree=1.0,
        pairwise_col_sampling=None,  # New parameter: float in (0, 1]
        random_state=None,
        warm_start=False,
    ):
        # Validate arguments
        self._validate_hyperparams(
            objective=objective,
            num_bins=num_bins,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            min_split_gain=min_split_gain,
            threads_per_block=threads_per_block,
            rows_per_thread=rows_per_thread,
            L2_reg=L2_reg,
            colsample_bytree=colsample_bytree,
            pairwise_col_sampling=pairwise_col_sampling,
        )

        self.objective = objective
        self.num_bins = num_bins
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.device = device
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.threads_per_block = threads_per_block
        self.rows_per_thread = rows_per_thread
        self.L2_reg = L2_reg
        self.colsample_bytree = colsample_bytree
        self.pairwise_col_sampling = pairwise_col_sampling
        self.random_state = random_state
        self.warm_start = warm_start
        
        # Internal state
        self.forest = [{} for _ in range(self.n_estimators)] if objective == "regression" else []
        self.bin_edges = None
        self.base_prediction = None
        self.unique_eras = None
        self.num_classes = None
        self.classes_ = None
        self.label_encoder = None
        self.feature_importance_ = None
        self.per_era_feature_importance_ = None
        self._is_fitted = False
        self._trees_trained = 0

    def _validate_hyperparams(self, **kwargs):
        if kwargs["objective"] not in ["regression", "multiclass", "binary"]:
            raise ValueError(f"objective must be 'regression', 'binary', or 'multiclass', got {kwargs['objective']}.")
        
        # Numeric checks
        if not (2 <= kwargs["num_bins"] <= 127):
            raise ValueError("num_bins must be between 2 and 127 inclusive.")
        if not (0.0 < kwargs["learning_rate"] <= 1.0):
            raise ValueError("learning_rate must be in (0.0, 1.0].")
        if kwargs["colsample_bytree"] <= 0 or kwargs["colsample_bytree"] > 1:
            raise ValueError(f"Invalid colsample_bytree: {kwargs['colsample_bytree']}. Must be in (0, 1].")
        
        pw_samp = kwargs.get("pairwise_col_sampling")
        if pw_samp is not None:
            if not isinstance(pw_samp, (float, int)) or not (0.0 < pw_samp <= 1.0):
                raise ValueError(f"pairwise_col_sampling must be None or in (0, 1], got {pw_samp}")

    def _compute_tree_predictions(self, tree, bin_indices):
        num_samples = bin_indices.size(0)
        predictions = torch.zeros(num_samples, device=self.device, dtype=torch.float32)
        
        def traverse(node, sample_mask):
            if "leaf_value" in node:
                predictions[sample_mask] = node["leaf_value"] * self.learning_rate
            else:
                feature_idx = node["feature"]
                split_bin = node["bin"]
                go_left = bin_indices[sample_mask, feature_idx] <= split_bin
                
                left_mask = sample_mask.clone()
                left_mask[sample_mask] = go_left
                right_mask = sample_mask.clone()
                right_mask[sample_mask] = ~go_left
                
                traverse(node["left"], left_mask)
                traverse(node["right"], right_mask)
        
        all_samples = torch.ones(num_samples, dtype=torch.bool, device=self.device)
        traverse(tree, all_samples)
        return predictions

    def preprocess_gpu_data(self, X_np, Y_np, era_id_np):
        with torch.no_grad():
            self.num_samples, self.num_features = X_np.shape
            Y_gpu = torch.from_numpy(Y_np).type(torch.float32).to(self.device)
            era_id_gpu = torch.from_numpy(era_id_np).type(torch.int32).to(self.device)

            bin_indices = torch.empty((self.num_samples, self.num_features), dtype=torch.int8, device=self.device)
            
            # Simplified binner logic for brevity
            bin_edges = torch.empty((self.num_features, self.num_bins - 1), dtype=torch.float32, device=self.device)
            for f in range(self.num_features):
                X_f = torch.as_tensor(X_np[:, f], device=self.device, dtype=torch.float32).contiguous()
                quantiles = torch.linspace(0, 1, self.num_bins + 1, device=self.device)[1:-1]
                bin_edges_f = torch.quantile(X_f, quantiles)
                node_kernel.custom_cuda_binner(X_f, bin_edges_f, bin_indices[:, f])
                bin_edges[f, :] = bin_edges_f

            unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)
            return bin_indices, era_indices, bin_edges, unique_eras, Y_gpu

    def compute_histograms(self, sample_indices, feature_indices):
        k = len(feature_indices)
        grad_hist = torch.zeros((self.num_eras, k, self.num_bins), device=self.device, dtype=torch.float32)
        hess_hist = torch.zeros((self.num_eras, k, self.num_bins), device=self.device, dtype=torch.float32)

        node_kernel.compute_histogram3(
            self.bin_indices, self.residual, sample_indices, feature_indices,
            self.era_indices, grad_hist, hess_hist, self.num_bins,
            self.threads_per_block, self.rows_per_thread
        )
        return grad_hist, hess_hist

    def find_best_split(self, gradient_histogram, hessian_histogram):
        # Placeholder for kernel call: update self.per_era_gain/direction
        node_kernel.compute_split(
            gradient_histogram, hessian_histogram, self.min_split_gain,
            self.min_child_weight, self.L2_reg, self.per_era_gain,
            self.per_era_direction, self.threads_per_block
        )

        if self.num_eras == 1:
            era_splitting_criterion = self.per_era_gain[0, :, :]
            dir_score_mask = era_splitting_criterion > self.min_split_gain
        else:
            directional_agreement = self.per_era_direction.mean(dim=0).abs()
            era_splitting_criterion = self.per_era_gain.mean(dim=0)
            dir_score_mask = (directional_agreement == directional_agreement.max()) & (era_splitting_criterion > self.min_split_gain)

        if not dir_score_mask.any():
            return -1, -1
        
        era_splitting_criterion[dir_score_mask == 0] = float("-inf")
        best_idx = torch.argmax(era_splitting_criterion)
        split_bins = self.num_bins - 1
        return (best_idx // split_bins).item(), (best_idx % split_bins).item()

    def grow_tree(self, grad_hist, hess_hist, node_indices, depth, class_k=None):
        if depth == self.max_depth:
            val = self.residual[node_indices].mean()
            if class_k is not None: self.gradients[node_indices, class_k] += self.learning_rate * val
            else: self.gradients[node_indices] += self.learning_rate * val
            return {"leaf_value": val.item(), "samples": node_indices.numel()}

        local_feat, best_bin = self.find_best_split(grad_hist, hess_hist)
        if local_feat == -1:
            val = self.residual[node_indices].mean()
            return {"leaf_value": val.item(), "samples": node_indices.numel()}

        global_feat = self.feat_indices_tree[local_feat].item()
        # Importance tracking
        self.per_era_feature_importance_[:, global_feat] += self.per_era_gain[:, local_feat, best_bin].cpu().numpy()

        split_mask = self.bin_indices[node_indices, global_feat] <= best_bin
        left_idx, right_idx = node_indices[split_mask], node_indices[~split_mask]

        # Histogram subtraction optimization
        if left_idx.numel() <= right_idx.numel():
            gl, hl = self.compute_histograms(left_idx, self.feat_indices_tree)
            gr, hr = grad_hist - gl, hess_hist - hl
        else:
            gr, hr = self.compute_histograms(right_idx, self.feat_indices_tree)
            gl, hl = grad_hist - gr, hess_hist - hr

        return {
            "feature": global_feat, "bin": best_bin,
            "left": self.grow_tree(gl, hl, left_idx, depth + 1, class_k),
            "right": self.grow_tree(gr, hr, right_idx, depth + 1, class_k)
        }

    def grow_forest(self):
        """Standard GBM loop with Pairwise Column Sampling."""
        if not hasattr(self, 'training_loss') or not self.warm_start or not self._is_fitted:
            self.training_loss, self.eval_loss = [], []
            self.per_era_feature_importance_ = np.zeros((self.num_eras, self.num_features), dtype=np.float32)
        
        start_iter = self._trees_trained if self.warm_start and self._is_fitted else 0
        
        for i in range(start_iter, self.n_estimators):
            self.residual = self.Y_gpu - self.gradients
            
            # --- Pairwise Column Sampling Logic ---
            if self.pairwise_col_sampling is not None:
                if self.num_features % 2 != 0:
                    raise ValueError("Pairwise sampling requires an even number of columns (n base + n derived).")
                n_base = self.num_features // 2
                k_base = max(1, int(self.pairwise_col_sampling * n_base))
                
                # Sample base indices, then add derived (index + n_base)
                base_idx = torch.randperm(n_base, device=self.device, dtype=torch.int32)[:k_base]
                self.feat_indices_tree = torch.cat([base_idx, base_idx + n_base])
            elif self.colsample_bytree < 1.0:
                k = max(1, int(self.colsample_bytree * self.num_features))
                self.feat_indices_tree = torch.randperm(self.num_features, device=self.device, dtype=torch.int32)[:k]
            else:
                self.feat_indices_tree = self.feature_indices

            k_current = self.feat_indices_tree.size(0)
            self.per_era_gain = torch.zeros((self.num_eras, k_current, self.num_bins-1), device=self.device)
            self.per_era_direction = torch.zeros((self.num_eras, k_current, self.num_bins-1), device=self.device)

            gh, hh = self.compute_histograms(self.root_node_indices, self.feat_indices_tree)
            self.forest[i] = self.grow_tree(gh, hh, self.root_node_indices, 0)
            self._trees_trained += 1

    def fit(self, X, y, era_id=None, **kwargs):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        if era_id is None: era_id = np.ones(X.shape[0], dtype="int32")
        
        self.bin_indices, self.era_indices, self.bin_edges, self.unique_eras, self.Y_gpu = self.preprocess_gpu_data(X, y, era_id)
        self.num_eras = len(self.unique_eras)
        self.gradients = torch.full_like(self.Y_gpu, self.Y_gpu.mean().item())
        self.root_node_indices = torch.arange(X.shape[0], device=self.device, dtype=torch.int32)
        self.feature_indices = torch.arange(X.shape[1], device=self.device, dtype=torch.int32)

        with torch.no_grad():
            self.grow_forest()
        
        self._is_fitted = True
        return self

    def predict(self, X):
        # Implementation of predict using self.forest and self.bin_edges
        pass