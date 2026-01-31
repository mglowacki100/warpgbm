import torch
import triton
import triton.language as tl


@triton.jit
def _bin_column_kernel(
    x_ptr, bin_edges_ptr, bin_indices_ptr,
    N, B_minus1,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    val = tl.load(x_ptr + offsets, mask=mask)

    bin_idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)


    for b in range(B_minus1):
        edge = tl.load(bin_edges_ptr + b)
        bin_idx = tl.where(val >= edge, b + 1, bin_idx)
    
    #tl.store(bin_indices_ptr + offsets, bin_idx, mask=mask)
    tl.store(bin_indices_ptr + offsets, bin_idx.to(tl.int8), mask=mask)


# --- KERNEL 2: HISTOGRAM ---
@triton.jit
def _histogram_kernel(
    bin_ptr, res_ptr, sample_idx_ptr, feat_idx_ptr, era_idx_ptr,
    grad_hist_ptr, hess_hist_ptr,
    N, F_master, F_active, B, num_eras,
    BLOCK_SIZE: tl.constexpr
):
    feat_order_idx = tl.program_id(0) # Która cecha z listy wybranych
    tile_idx = tl.program_id(1)      # Który blok próbek
    
    # Pobierz prawdziwy indeks cechy z mapowania
    feat_idx = tl.load(feat_idx_ptr + feat_order_idx)
    
    row_offsets = tile_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets < N
    
    # Załadowanie indeksów próbek i przypisanie do er oraz binów
    s_idx = tl.load(sample_idx_ptr + row_offsets, mask=row_mask)
    era = tl.load(era_idx_ptr + s_idx, mask=row_mask)
    # Załadowanie binów: bin_indices[sample * F_master + feat]
    b_val = tl.load(bin_ptr + s_idx * F_master + feat_idx, mask=row_mask)
    res = tl.load(res_ptr + s_idx, mask=row_mask)

    # Iteracja po erach i binach wewnątrz bloku
    # Triton zoptymalizuje to pod kątem dostępu do pamięci
    for e in range(num_eras):
        era_mask = row_mask & (era == e)
        for b in range(B):
            final_mask = era_mask & (b_val == b)
            
            g_sum = tl.sum(tl.where(final_mask, res, 0.0), axis=0)
            h_sum = tl.sum(tl.where(final_mask, 1.0, 0.0), axis=0)
            
            if g_sum != 0 or h_sum != 0:
                out_off = e * (F_active * B) + feat_order_idx * B + b
                tl.atomic_add(grad_hist_ptr + out_off, g_sum)
                tl.atomic_add(hess_hist_ptr + out_off, h_sum)

# --- KERNEL 3: DIRECTIONAL SPLIT ---
@triton.jit
def _split_kernel(
    G_ptr, H_ptr, gain_ptr, dir_ptr,
    E, F, B,
    min_child_samples, eps,
    BLOCK_SIZE: tl.constexpr
):
    f_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    e_idx = tl.program_id(1)
    
    f_mask = f_idx < F
    
    # Przesunięcia bazowe
    base = e_idx * F * B + f_idx * B
    base_out = e_idx * F * (B - 1) + f_idx * (B - 1)

    # G_total i H_total
    g_tot = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    h_tot = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for b in range(B):
        g_tot += tl.load(G_ptr + base + b, mask=f_mask, other=0.0)
        h_tot += tl.load(H_ptr + base + b, mask=f_mask, other=0.0)

    g_left = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    h_left = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for b in range(B - 1):
        g_left += tl.load(G_ptr + base + b, mask=f_mask, other=0.0)
        h_left += tl.load(H_ptr + base + b, mask=f_mask, other=0.0)
        
        g_right = g_tot - g_left
        h_right = h_tot - h_left
        
        valid = (h_left >= min_child_samples) & (h_right >= min_child_samples)
        
        gain = (g_left * g_left) / (h_left + eps) + \
               (g_right * g_right) / (h_right + eps) - \
               (g_tot * g_tot) / (h_tot + eps)
        
        direction = tl.where((g_left / (h_left + eps)) > (g_right / (h_right + eps)), 1.0, -1.0)
        
        tl.store(gain_ptr + base_out + b, tl.where(valid, gain, 0.0), mask=f_mask)
        tl.store(dir_ptr + base_out + b, tl.where(valid, direction, 0.0), mask=f_mask)


@triton.jit
def _predict_kernel(
    bin_ptr, tree_ptr, out_ptr,
    N, F, T, max_nodes, lr,
    BLOCK_SIZE: tl.constexpr
):
    block_size: tl.constexpr = BLOCK_SIZE  # konieczne
    pid = tl.program_id(0)
    idx = pid * block_size + tl.arange(0, block_size)
    
    mask = idx < (N * T)
    s_idx = idx % N
    t_idx = idx // N

    node_id = tl.zeros([block_size], dtype=tl.int32)
    active = mask

    for _ in range(64):
        if not tl.reduce(active, 0, "any"):
            break

        tree_node_base = t_idx * max_nodes * 6 + node_id * 6
        is_leaf = tl.load(tree_ptr + tree_node_base + 4, mask=active)
        leaf_mask = active & (is_leaf > 0.5)
        
        if tl.reduce(leaf_mask, 0, "any"):
            val = tl.load(tree_ptr + tree_node_base + 5, mask=leaf_mask)
            tl.atomic_add(out_ptr + s_idx, val * lr, mask=leaf_mask)
            active = active & (~leaf_mask)

        feat = tl.load(tree_ptr + tree_node_base + 0, mask=active).to(tl.int32)
        split_bin = tl.load(tree_ptr + tree_node_base + 1, mask=active)
        bin_val = tl.load(bin_ptr + s_idx * F + feat, mask=active)

        left_id = tl.load(tree_ptr + tree_node_base + 2, mask=active).to(tl.int32)
        right_id = tl.load(tree_ptr + tree_node_base + 3, mask=active).to(tl.int32)
        node_id = tl.where(bin_val <= split_bin, left_id, right_id)



# --- WRAPPERS ---

def custom_cuda_binner(X, bin_edges, bin_indices):
    N = X.size(0)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    _bin_column_kernel[grid](X, bin_edges, bin_indices, N, bin_edges.size(0), BLOCK_SIZE=256)

def compute_histogram3(bin_indices, residuals, sample_indices, feature_indices, era_indices, 
                       grad_hist, hess_hist, num_bins, threads_per_block=256, rows_per_thread=1):
    N = sample_indices.size(0)
    F_active = feature_indices.size(0)
    F_master = bin_indices.size(1)
    num_eras = grad_hist.size(0)
    
    # Czyścimy histogramy przed użyciem
    grad_hist.fill_(0)
    hess_hist.fill_(0)

    grid = (F_active, triton.cdiv(N, threads_per_block))
    _histogram_kernel[grid](
        bin_indices, residuals, sample_indices, feature_indices, era_indices,
        grad_hist, hess_hist, N, F_master, F_active, num_bins, num_eras,
        BLOCK_SIZE=threads_per_block
    )

def compute_split(G, H, min_split_gain, min_child_samples, eps, per_era_gain, per_era_direction, threads=128):
    E, F, B = G.shape
    grid = (triton.cdiv(F, threads), E)
    _split_kernel[grid](
        G, H, per_era_gain, per_era_direction, E, F, B,
        min_child_samples, eps, BLOCK_SIZE=threads
    )

def predict_forest(bin_indices, tree_tensor, learning_rate, out):
    N, F = bin_indices.shape
    T, max_nodes, _ = tree_tensor.shape
    grid = lambda meta: (triton.cdiv(N * T, meta['BLOCK_SIZE']),)
    _predict_kernel[grid](
        bin_indices, tree_tensor, out, N, F, T, max_nodes, learning_rate, BLOCK_SIZE=512
    )