import torch
import taichi as ti

# Initialize Taichi with CUDA backend for GPU acceleration
ti.init(arch=ti.cuda)

# --- KERNEL 1: BINNING KERNEL ---
@ti.kernel
def bin_column_kernel(
    x: ti.types.ndarray(ndim=1, dtype=ti.f32),  # Input values [N]
    bin_edges: ti.types.ndarray(ndim=1, dtype=ti.f32),  # Bin edges [B]
    bin_indices: ti.types.ndarray(ndim=1, dtype=ti.i8),  # Output bin indices [N]
    N: ti.i32,
    B_minus1: ti.i32
):
    for i in ti.ndrange(N):
        val = x[i]
        bin_idx = 0
        for b in range(B_minus1):
            edge = bin_edges[b]
            if val >= edge:
                bin_idx = b + 1
        bin_indices[i] = bin_idx

# --- KERNEL 2: HISTOGRAM KERNEL ---
@ti.kernel
def histogram_kernel(
    bin_ptr: ti.types.ndarray(ndim=2, dtype=ti.i8),  # Bin indices [N, F_master]
    res_ptr: ti.types.ndarray(ndim=1, dtype=ti.f32),  # Residuals [N]
    sample_idx_ptr: ti.types.ndarray(ndim=1, dtype=ti.i32),  # Active sample indices [N_active]
    feat_idx_ptr: ti.types.ndarray(ndim=1, dtype=ti.i32),  # Active feature mapping [F_active]
    era_idx_ptr: ti.types.ndarray(ndim=1, dtype=ti.i32),  # Era indices [N]
    grad_hist_ptr: ti.types.ndarray(ndim=3, dtype=ti.f32),  # Output: Gradient histogram [num_eras, F_active, B]
    hess_hist_ptr: ti.types.ndarray(ndim=3, dtype=ti.f32),  # Output: Hessian histogram [num_eras, F_active, B]
    N: ti.i32,  # Total active samples
    F_master: ti.i32,  # Total features in original dataset
    F_active: ti.i32,  # Number of features currently processing
    B: ti.i32,  # Number of bins
    num_eras: ti.i32   # Number of eras
):
    # Parallelize over features and samples
    for feat_order_idx in ti.ndrange(F_active):
        feat_idx = feat_idx_ptr[feat_order_idx]
        for row_idx in ti.ndrange(N):
            s_idx = sample_idx_ptr[row_idx]
            era = era_idx_ptr[s_idx]
            bin_val = bin_ptr[s_idx, feat_idx]
            residual = res_ptr[s_idx]
            
            # Atomic add for gradient and hessian (Taichi handles atomics efficiently)
            ti.atomic_add(grad_hist_ptr[era, feat_order_idx, bin_val], residual)
            ti.atomic_add(hess_hist_ptr[era, feat_order_idx, bin_val], 1.0)

# --- KERNEL 3: DIRECTIONAL SPLIT KERNEL ---
@ti.kernel
def split_kernel(
    G_ptr: ti.types.ndarray(ndim=3, dtype=ti.f32),  # Gradients [E, F, B]
    H_ptr: ti.types.ndarray(ndim=3, dtype=ti.f32),  # Hessians [E, F, B]
    gain_ptr: ti.types.ndarray(ndim=3, dtype=ti.f32),  # Gains [E, F, B-1]
    dir_ptr: ti.types.ndarray(ndim=3, dtype=ti.f32),  # Directions [E, F, B-1]
    E: ti.i32,  # Number of eras
    F: ti.i32,  # Number of features
    B: ti.i32,  # Number of bins
    min_child_samples: ti.i32,
    eps: ti.f32
):
    for e_idx in ti.ndrange(E):
        for f_idx in ti.ndrange(F):
            # Compute total G and H
            g_tot = 0.0
            h_tot = 0.0
            for b in range(B):
                g_tot += G_ptr[e_idx, f_idx, b]
                h_tot += H_ptr[e_idx, f_idx, b]
            
            g_left = 0.0
            h_left = 0.0
            for b in range(B - 1):
                g_left += G_ptr[e_idx, f_idx, b]
                h_left += H_ptr[e_idx, f_idx, b]
                
                g_right = g_tot - g_left
                h_right = h_tot - h_left
                
                valid = (h_left >= min_child_samples) and (h_right >= min_child_samples)
                
                gain = (g_left * g_left) / (h_left + eps) + \
                       (g_right * g_right) / (h_right + eps) - \
                       (g_tot * g_tot) / (h_tot + eps)
                
                direction = 1.0 if (g_left / (h_left + eps)) > (g_right / (h_right + eps)) else -1.0
                
                gain_ptr[e_idx, f_idx, b] = gain if valid else 0.0
                dir_ptr[e_idx, f_idx, b] = direction if valid else 0.0

# --- KERNEL 4: PREDICT KERNEL ---
@ti.kernel
def predict_kernel(
    bin_ptr: ti.types.ndarray(ndim=2, dtype=ti.i8),  # Bin indices [N, F]
    tree_ptr: ti.types.ndarray(ndim=3, dtype=ti.f32),  # Tree tensor [T, max_nodes, 6]
    out_ptr: ti.types.ndarray(ndim=1, dtype=ti.f32),  # Output predictions [N]
    N: ti.i32,  # Number of samples
    F: ti.i32,  # Number of features
    T: ti.i32,  # Number of trees
    max_nodes: ti.i32,  # Max nodes per tree
    lr: ti.f32   # Learning rate
):
    for idx in ti.ndrange(N * T):
        s_idx = idx % N  # Sample index
        t_idx = idx // N  # Tree index
        
        node_id = 0
        active = True
        
        for _ in range(64):  # Max depth assumption
            if not active:
                break
            
            tree_node_base = t_idx * max_nodes * 6 + node_id * 6
            
            is_leaf = tree_ptr[t_idx, node_id, 4] > 0.5
            
            if is_leaf:
                leaf_val = tree_ptr[t_idx, node_id, 5]
                ti.atomic_add(out_ptr[s_idx], leaf_val * lr)
                active = False
            else:
                feat = ti.cast(tree_ptr[t_idx, node_id, 0], ti.i32)
                split_bin = tree_ptr[t_idx, node_id, 1]
                bin_val = bin_ptr[s_idx, feat]
                
                left_id = ti.cast(tree_ptr[t_idx, node_id, 2], ti.i32)
                right_id = ti.cast(tree_ptr[t_idx, node_id, 3], ti.i32)
                
                node_id = left_id if bin_val <= split_bin else right_id

# --- WRAPPERS (Adapted for Taichi) ---
def custom_cuda_binner(X, bin_edges, bin_indices):
    # Assuming X, bin_edges, bin_indices are Torch tensors; convert to Taichi ndarrays if needed
    N = X.size(0)
    B_minus1 = bin_edges.size(0)
    bin_column_kernel(X, bin_edges, bin_indices, N, B_minus1)

def compute_histogram3(bin_indices, residuals, sample_indices, feature_indices, era_indices,
                       grad_hist, hess_hist, num_bins,
                       threads_per_block=None, rows_per_thread=None, #just to make signature compliant with core.py
                       ):
    N = sample_indices.size(0)
    F_active = feature_indices.size(0)
    F_master = bin_indices.size(1)
    num_eras = grad_hist.size(0)
    
    # Clear histograms (Taichi ndarrays are zero-initialized, but to be safe)
    grad_hist.fill(0.0)
    hess_hist.fill(0.0)
    
    histogram_kernel(
        bin_indices, residuals, sample_indices, feature_indices, era_indices,
        grad_hist, hess_hist, N, F_master, F_active, num_bins, num_eras
    )

def compute_split(G, H, min_split_gain, min_child_samples, eps, per_era_gain, per_era_direction):
    E, F, B = G.shape
    split_kernel(
        G, H, per_era_gain, per_era_direction, E, F, B,
        min_child_samples, eps
    )

def predict_forest(bin_indices, tree_tensor, learning_rate, out):
    N, F = bin_indices.shape
    T, max_nodes, _ = tree_tensor.shape
    out.fill(0.0)  # Initialize output
    predict_kernel(
        bin_indices, tree_tensor, out, N, F, T, max_nodes, learning_rate
    )