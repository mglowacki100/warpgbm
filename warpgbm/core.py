import torch
import numpy as np
import gc
import cutile
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any, Optional, List

# --- CUDA Kernels via Cutile JIT ---

CUDA_SRC = """
extern "C" {
    // Quantile binning kernel
    __global__ void binner_kernel(
        const float* __restrict__ X, 
        const float* __restrict__ edges, 
        int8_t* __restrict__ out, 
        int num_samples, 
        int num_bins_minus_one) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_samples) {
            float val = X[idx];
            int bin = 0;
            for (int i = 0; i < num_bins_minus_one; ++i) {
                if (val > edges[i]) bin++;
                else break;
            }
            out[idx] = (int8_t)bin;
        }
    }

    // Gradient and Hessian Histogram kernel
    __global__ void histogram_kernel(
        const int8_t* __restrict__ bin_indices,
        const float* __restrict__ residuals,
        const float* __restrict__ hessians,
        const int* __restrict__ sample_indices,
        const int* __restrict__ feature_indices,
        const int* __restrict__ era_indices,
        float* grad_hist,
        float* hess_hist,
        int num_samples_in_node, 
        int num_features_subset, 
        int num_bins, 
        int total_features) {
        
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < num_samples_in_node) {
            int sample_idx = sample_indices[tid];
            int era = era_indices[sample_idx];
            float g = residuals[sample_idx];
            float h = hessians[sample_idx];

            for (int f = 0; f < num_features_subset; ++f) {
                int feat_idx = feature_indices[f];
                int bin = bin_indices[sample_idx * total_features + feat_idx];
                
                int hist_idx = (era * num_features_subset * num_bins) + (f * num_bins) + bin;
                atomicAdd(&grad_hist[hist_idx], g);
                atomicAdd(&hess_hist[hist_idx], h);
            }
        }
    }
}
"""

class WarpGBM(BaseEstimator):
    def __init__(
        self,
        objective="regression", # "regression", "binary", "multiclass"
        num_bins=32,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=20.0,
        min_split_gain=0.0,
        L2_reg=1e-3,
        colsample_bytree=1.0,
        device="cuda",
        random_state=None,
        early_stopping_rounds=None
    ):
        self.objective = objective
        self.num_bins = num_bins
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.L2_reg = L2_reg
        self.colsample_bytree = colsample_bytree
        self.device = device
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        
        # Internals
        self.forest = []
        self._is_fitted = False
        self.bin_edges = None
        self.label_encoder = None
        self.num_classes = 1
        self._cuda_mod = cutile.compile_cuda(CUDA_SRC)

    def _get_gradients_hessians(self, y_true, y_pred):
        """Compute gradients and hessians based on objective."""
        if self.objective == "regression":
            # MSE Loss: Grad = (pred - true), Hess = 1.0
            grad = y_pred - y_true
            hess = torch.ones_like(grad)
        elif self.objective == "binary":
            # LogLoss: Grad = (p - y), Hess = p * (1 - p)
            prob = torch.sigmoid(y_pred)
            grad = prob - y_true
            hess = prob * (1.0 - prob)
        elif self.objective == "multiclass":
            # Softmax: Grad = (p_k - y_k), Hess = p_k * (1 - p_k)
            probs = torch.softmax(y_pred, dim=1)
            y_onehot = torch.zeros_like(probs)
            y_onehot.scatter_(1, y_true.unsqueeze(1).long(), 1.0)
            grad = probs - y_onehot
            hess = probs * (1.0 - probs)
        return grad, hess

    def preprocess_data(self, X_np, y_np, era_id_np):
        num_samples, num_features = X_np.shape
        X_gpu = torch.from_numpy(X_np).float().to(self.device)
        y_gpu = torch.from_numpy(y_np).float().to(self.device)
        era_id_gpu = torch.from_numpy(era_id_np).int().to(self.device)

        bin_indices = torch.empty((num_samples, num_features), dtype=torch.int8, device=self.device)
        bin_edges = []

        for f in range(num_features):
            feat_col = X_gpu[:, f].contiguous()
            q = torch.linspace(0, 1, self.num_bins + 1, device=self.device)[1:-1]
            edges = torch.quantile(feat_col, q)
            bin_edges.append(edges)
            
            grid = (num_samples + 63) // 64
            self._cuda_mod.binner_kernel(grid, 64)(
                feat_col, edges, bin_indices[:, f], num_samples, self.num_bins - 1
            )
        
        self.bin_edges = torch.stack(bin_edges)
        unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)
        return bin_indices, era_indices, unique_eras, y_gpu

    def compute_histograms(self, sample_indices, feat_indices, residuals, hessians):
        num_eras = len(self.unique_eras)
        num_feat_subset = len(feat_indices)
        
        gh = torch.zeros((num_eras, num_feat_subset, self.num_bins), device=self.device)
        hh = torch.zeros((num_eras, num_feat_subset, self.num_bins), device=self.device)

        grid = (sample_indices.numel() + 63) // 64
        self._cuda_mod.histogram_kernel(grid, 64)(
            self.bin_indices, residuals, hessians, sample_indices, feat_indices,
            self.era_indices, gh, hh, sample_indices.numel(), num_feat_subset, self.num_bins, self.num_features
        )
        return gh, hh

    def find_best_split(self, gh, hh):
        # Prefix sums for split points
        GL = torch.cumsum(gh, dim=2)[:, :, :-1]
        HL = torch.cumsum(hh, dim=2)[:, :, :-1]
        
        G_tot = gh.sum(dim=2, keepdim=True)
        H_tot = hh.sum(dim=2, keepdim=True)
        
        GR = G_tot - GL
        HR = H_tot - HL

        # Gain calculation with L2
        gain = (GL**2 / (HL + self.L2_reg)) + (GR**2 / (HR + self.L2_reg)) - (G_tot**2 / (H_tot + self.L2_reg))
        
        # WarpGBM Heuristic: Directional Agreement across eras
        # Ensures that a split is robust across different time periods/groups
        direction = torch.sign(GL / (HL + 1e-7) - GR / (HR + 1e-7))
        
        if len(self.unique_eras) > 1:
            avg_gain = gain.mean(dim=0)
            agreement = direction.mean(dim=0).abs()
            mask = (agreement > 0.4) & (avg_gain > self.min_split_gain)
        else:
            avg_gain = gain[0]
            mask = avg_gain > self.min_split_gain

        if not mask.any(): return -1, -1

        avg_gain[~mask] = -1e10
        best_idx = torch.argmax(avg_gain)
        return (best_idx // (self.num_bins - 1)).item(), (best_idx % (self.num_bins - 1)).item()

    def grow_tree(self, gh, hh, node_indices, depth, current_res, current_hess):
        # Stop criteria
        if depth >= self.max_depth or node_indices.numel() < self.min_child_weight:
            leaf_val = - (current_res.sum() / (current_hess.sum() + self.L2_reg))
            return {"leaf": leaf_val.item()}

        l_feat, b_idx = self.find_best_split(gh, hh)
        if l_feat == -1:
            leaf_val = - (current_res.sum() / (current_hess.sum() + self.L2_reg))
            return {"leaf": leaf_val.item()}

        g_feat = self.feat_subset[l_feat]
        mask = self.bin_indices[node_indices, g_feat] <= b_idx
        
        left_idx, right_idx = node_indices[mask], node_indices[~mask]
        
        # Optimization: calculate smaller hist, subtract for larger
        if left_idx.numel() < right_idx.numel():
            gh_l, hh_l = self.compute_histograms(left_idx, self.feat_subset, current_res[mask], current_hess[mask])
            gh_r, hh_r = gh - gh_l, hh - hh_l
        else:
            gh_r, hh_r = self.compute_histograms(right_idx, self.feat_subset, current_res[~mask], current_hess[~mask])
            gh_l, hh_l = gh - gh_r, hh - hh_r

        return {
            "f": g_feat.item(), "b": b_idx,
            "left": self.grow_tree(gh_l, hh_l, left_idx, depth+1, current_res[mask], current_hess[mask]),
            "right": self.grow_tree(gh_r, hh_r, right_idx, depth+1, current_res[~mask], current_hess[~mask])
        }

    def fit(self, X, y, era_id=None):
        if self.objective in ["binary", "multiclass"]:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            self.num_classes = len(self.label_encoder.classes_)
        
        if era_id is None: era_id = np.zeros(X.shape[0])
        
        self.bin_indices, self.era_indices, self.unique_eras, self.y_gpu = self.preprocess_data(X, y, era_id)
        self.num_samples, self.num_features = X.shape
        
        # Prediction shape
        if self.objective == "multiclass":
            self.y_pred = torch.zeros((self.num_samples, self.num_classes), device=self.device)
        else:
            self.y_pred = torch.zeros(self.num_samples, device=self.device)
            
        root_indices = torch.arange(self.num_samples, device=self.device, dtype=torch.int32)
        all_feats = torch.arange(self.num_features, device=self.device, dtype=torch.int32)

        for i in range(self.n_estimators):
            # Multiclass: train one tree per class per iteration
            round_trees = []
            for k in range(self.num_classes if self.objective == "multiclass" else 1):
                grad, hess = self._get_gradients_hessians(self.y_gpu, self.y_pred)
                
                # Extract residuals for current class/round
                curr_grad = grad[:, k] if self.objective == "multiclass" else grad
                curr_hess = hess[:, k] if self.objective == "multiclass" else hess

                if self.colsample_bytree < 1.0:
                    self.feat_subset = torch.randperm(self.num_features, device=self.device)[:int(self.num_features*self.colsample_bytree)].int()
                else:
                    self.feat_subset = all_feats

                gh, hh = self.compute_histograms(root_indices, self.feat_subset, curr_grad, curr_hess)
                tree = self.grow_tree(gh, hh, root_indices, 0, curr_grad, curr_hess)
                round_trees.append(tree)
                
                # Update predictions
                update = self._apply_tree(tree, self.bin_indices)
                if self.objective == "multiclass":
                    self.y_pred[:, k] += self.learning_rate * update
                else:
                    self.y_pred += self.learning_rate * update

            self.forest.append(round_trees)
            if (i+1) % 5 == 0:
                print(f"Iteration {i+1} completed")

        self._is_fitted = True
        return self

    def _apply_tree(self, tree, bins):
        if "leaf" in tree:
            return torch.full((bins.shape[0],), tree["leaf"], device=self.device)
        mask = bins[:, tree["f"]] <= tree["b"]
        res = torch.empty(bins.shape[0], device=self.device)
        res[mask] = self._apply_tree(tree["left"], bins[mask])
        res[~mask] = self._apply_tree(tree["right"], bins[~mask])
        return res

    def predict_raw(self, X):
        num_s = X.shape[0]
        X_g = torch.from_numpy(X).float().to(self.device)
        bins = torch.empty((num_s, self.num_features), dtype=torch.int8, device=self.device)
        
        for f in range(self.num_features):
            grid = (num_s + 63) // 64
            self._cuda_mod.binner_kernel(grid, 64)(X_g[:,f], self.bin_edges[f], bins[:,f], num_s, self.num_bins - 1)
        
        if self.objective == "multiclass":
            preds = torch.zeros((num_s, self.num_classes), device=self.device)
            for round_trees in self.forest:
                for k, tree in enumerate(round_trees):
                    preds[:, k] += self.learning_rate * self._apply_tree(tree, bins)
        else:
            preds = torch.zeros(num_s, device=self.device)
            for round_trees in self.forest:
                preds += self.learning_rate * self._apply_tree(round_trees[0], bins)
        return preds

    def predict(self, X):
        raw = self.predict_raw(X)
        if self.objective == "regression":
            return raw.cpu().numpy()
        elif self.objective == "binary":
            return (torch.sigmoid(raw) > 0.5).cpu().numpy()
        else:
            return self.label_encoder.inverse_transform(torch.argmax(raw, dim=1).cpu().numpy())

    def predict_proba(self, X):
        raw = self.predict_raw(X)
        if self.objective == "binary":
            p = torch.sigmoid(raw).cpu().numpy()
            return np.vstack([1-p, p]).T
        return torch.softmax(raw, dim=1).cpu().numpy()