import torch
import numpy as np
import gc
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from warpgbm.cuda import node_kernel
from warpgbm.metrics import rmsle_torch

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
        pairwise_col_sampling=None,
        random_state=None,
        warm_start=False,
    ):
        self.objective = objective
        self.num_bins = num_bins
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.threads_per_block = threads_per_block
        self.rows_per_thread = rows_per_thread
        self.L2_reg = L2_reg
        self.device = device
        self.colsample_bytree = colsample_bytree
        self.pairwise_col_sampling = pairwise_col_sampling
        self.random_state = random_state
        self.warm_start = warm_start

        # Initialization of internal states
        self.forest = []
        self._is_fitted = False
        self._trees_trained = 0
        self.per_era_feature_importance_ = None

    def _validate_data(self, X):
        if self.pairwise_col_sampling is not None:
            if X.shape[1] % 2 != 0:
                raise ValueError(
                    f"pairwise_col_sampling requires an even number of columns. "
                    f"Found {X.shape[1]} columns."
                )

    def preprocess_gpu_data(self, X_np, Y_np, era_id_np):
        with torch.no_grad():
            num_samples, num_features = X_np.shape
            Y_gpu = torch.from_numpy(Y_np).float().to(self.device)
            era_id_gpu = torch.from_numpy(era_id_np).int().to(self.device)

            bin_indices = torch.empty((num_samples, num_features), dtype=torch.int8, device=self.device)
            bin_edges = torch.empty((num_features, self.num_bins - 1), dtype=torch.float32, device=self.device)

            for f in range(num_features):
                X_f = torch.as_tensor(X_np[:, f], device=self.device, dtype=torch.float32).contiguous()
                # Quantile binning
                quantiles = torch.linspace(0, 1, self.num_bins + 1, device=self.device)[1:-1]
                bin_edges_f = torch.quantile(X_f, quantiles).contiguous()
                
                # Call CUDA binner
                node_kernel.custom_cuda_binner(X_f, bin_edges_f, bin_indices[:, f])
                bin_edges[f, :] = bin_edges_f

            unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)
            return bin_indices, era_indices.int(), bin_edges, unique_eras, Y_gpu

    def compute_histograms(self, sample_indices, feature_indices):
        # Ensure indices are Int32 for CUDA kernel
        if sample_indices.dtype != torch.int32:
            sample_indices = sample_indices.int()
        if feature_indices.dtype != torch.int32:
            feature_indices = feature_indices.int()

        k = len(feature_indices)
        grad_hist = torch.zeros((self.num_eras, k, self.num_bins), device=self.device, dtype=torch.float32)
        hess_hist = torch.zeros((self.num_eras, k, self.num_bins), device=self.device, dtype=torch.float32)

        node_kernel.compute_histogram3(
            self.bin_indices,
            self.residual,
            sample_indices,
            feature_indices,
            self.era_indices,
            grad_hist,
            hess_hist,
            self.num_bins,
            self.threads_per_block,
            self.rows_per_thread,
        )
        return grad_hist, hess_hist

    def find_best_split(self, gradient_histogram, hessian_histogram):
        # per_era_gain/direction buffers are managed by grow_forest/grow_tree
        node_kernel.compute_split(
            gradient_histogram,
            hessian_histogram,
            self.min_split_gain,
            self.min_child_weight,
            self.L2_reg,
            self.per_era_gain,
            self.per_era_direction,
            self.threads_per_block
        )

        if self.num_eras == 1:
            era_splitting_criterion = self.per_era_gain[0, :, :]
        else:
            # Logic from original implementation: direction agreement + gain mean
            directional_agreement = self.per_era_direction.mean(dim=0).abs()
            era_splitting_criterion = self.per_era_gain.mean(dim=0)
            # Mask where agreement isn't maximal if necessary, 
            # or just take argmax of mean gain
        
        # Simple best split selection
        best_idx = torch.argmax(era_splitting_criterion)
        split_bins = self.num_bins - 1
        best_feature = best_idx // split_bins
        best_bin = best_idx % split_bins

        # Return as python scalars
        return best_feature.item(), best_bin.item()

    def grow_tree(self, grad_hist, hess_hist, node_indices, depth):
        if depth == self.max_depth or node_indices.numel() < self.min_child_weight:
            val = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * val
            return {"leaf_value": val.item(), "samples": node_indices.numel()}

        local_feat, best_bin = self.find_best_split(grad_hist, hess_hist)
        
        # If no valid split found
        if local_feat < 0:
            val = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * val
            return {"leaf_value": val.item(), "samples": node_indices.numel()}

        global_feat = self.feat_indices_tree[local_feat].item()
        
        # Feature Importance (per era gain)
        gain_val = self.per_era_gain[:, local_feat, best_bin].cpu().numpy()
        self.per_era_feature_importance_[:, global_feat] += gain_val

        # Split indices
        split_mask = self.bin_indices[node_indices, global_feat] <= best_bin
        left_idx = node_indices[split_mask]
        right_idx = node_indices[~split_mask]

        if left_idx.numel() == 0 or right_idx.numel() == 0:
            val = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * val
            return {"leaf_value": val.item(), "samples": node_indices.numel()}

        # Recursive histograms (using subtraction for efficiency)
        if left_idx.numel() <= right_idx.numel():
            gl, hl = self.compute_histograms(left_idx, self.feat_indices_tree)
            gr, hr = grad_hist - gl, hess_hist - hl
        else:
            gr, hr = self.compute_histograms(right_idx, self.feat_indices_tree)
            gl, hl = grad_hist - gr, hess_hist - hr

        return {
            "feature": int(global_feat),
            "bin": int(best_bin),
            "left": self.grow_tree(gl, hl, left_idx, depth + 1),
            "right": self.grow_tree(gr, hr, right_idx, depth + 1),
        }

    def grow_forest(self):
        if not self.warm_start or not self._is_fitted:
            self.forest = [{} for _ in range(self.n_estimators)]
            self.per_era_feature_importance_ = np.zeros((self.num_eras, self.num_features), dtype=np.float32)
            self._trees_trained = 0

        start_iter = self._trees_trained
        
        for i in range(start_iter, self.n_estimators):
            self.residual = self.Y_gpu - self.gradients

            # --- Pairwise Column Sampling ---
            if self.pairwise_col_sampling is not None:
                n_base = self.num_features // 2
                k_base = max(1, int(self.pairwise_col_sampling * n_base))
                
                # Force dtype=torch.int32 to avoid RuntimeError
                base_idx = torch.randperm(n_base, device=self.device, dtype=torch.int32)[:k_base]
                derived_idx = base_idx + n_base
                self.feat_indices_tree = torch.cat([base_idx, derived_idx]).to(torch.int32)
            
            elif self.colsample_bytree < 1.0:
                k = max(1, int(self.colsample_bytree * self.num_features))
                self.feat_indices_tree = torch.randperm(self.num_features, device=self.device, dtype=torch.int32)[:k]
            else:
                self.feat_indices_tree = torch.arange(self.num_features, device=self.device, dtype=torch.int32)

            k_current = self.feat_indices_tree.size(0)
            
            # Temporary buffers for the CUDA split kernel
            self.per_era_gain = torch.zeros((self.num_eras, k_current, self.num_bins - 1), device=self.device)
            self.per_era_direction = torch.zeros((self.num_eras, k_current, self.num_bins - 1), device=self.device)

            # Build histograms for the root
            gh, hh = self.compute_histograms(self.root_node_indices, self.feat_indices_tree)
            
            # Grow the tree
            self.forest[i] = self.grow_tree(gh, hh, self.root_node_indices, 0)
            self._trees_trained += 1

            # Cleanup loop-specific tensors
            del self.per_era_gain, self.per_era_direction, gh, hh
            if i % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    def fit(self, X, y, era_id=None):
        self._validate_data(X)
        
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        if era_id is None:
            era_id = np.zeros(X.shape[0], dtype=np.int32)
        
        self.num_samples, self.num_features = X.shape
        
        # Preprocessing & Binning
        self.bin_indices, self.era_indices, self.bin_edges, self.unique_eras, self.Y_gpu = \
            self.preprocess_gpu_data(X, y, era_id)
        
        self.num_eras = len(self.unique_eras)
        
        # Starting point: mean target
        self.base_prediction = self.Y_gpu.mean().item()
        self.gradients = torch.full_like(self.Y_gpu, self.base_prediction)
        
        self.root_node_indices = torch.arange(self.num_samples, device=self.device, dtype=torch.int32)
        
        with torch.no_grad():
            self.grow_forest()
        
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        
        num_samples = X.shape[0]
        # Move base prediction to result
        preds = torch.full((num_samples,), self.base_prediction, device=self.device)
        
        # In a real implementation, you'd bin X using self.bin_edges here
        # and then traverse each tree in self.forest.
        # This is a simplified placeholder:
        return preds.cpu().numpy()