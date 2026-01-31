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
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['N']
)
@triton.jit
def _histogram_kernel(
    bin_ptr,  # Pointer to bin indices [N, F_master]
    res_ptr,  # Pointer to residuals [N]
    sample_idx_ptr,  # Pointer to active sample indices [N_active]
    feat_idx_ptr,  # Pointer to active feature mapping [F_active]
    era_idx_ptr,  # Pointer to era indices [N]
    grad_hist_ptr,  # Output: Gradient histogram [num_eras, F_active, B]
    hess_hist_ptr,  # Output: Hessian histogram [num_eras, F_active, B]
    N,  # Total number of active samples
    F_master,  # Total features in the original dataset
    F_active,  # Number of features we are currently processing
    B,  # Number of bins (e.g., 5)
    num_eras,  # Number of eras
    BLOCK_SIZE: tl.constexpr,
    ERA_BIN_SIZE: tl.constexpr  # Precomputed num_eras * B for flat shared memory
):
    # Thread Mapping
    # program_id(0): feature order index (0 to F_active-1)
    # program_id(1): tile/block of samples
    feat_order_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)
    
    # Load the actual feature index from the mapping table
    feat_idx = tl.load(feat_idx_ptr + feat_order_idx)
    
    # Calculate sample offsets for this block
    row_offsets = tile_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets < N
    
    # Data Loading (Coalesced Memory Access)
    # Load indices of samples belonging to the current node/active set
    s_indices = tl.load(sample_idx_ptr + row_offsets, mask=row_mask)
    
    # Load era and bin assignments for these specific samples
    eras = tl.load(era_idx_ptr + s_indices, mask=row_mask)
    bins = tl.load(bin_ptr + s_indices * F_master + feat_idx, mask=row_mask)
    residuals = tl.load(res_ptr + s_indices, mask=row_mask)
    
    # Allocate shared memory for local histograms (flat arrays for grad and hess)
    # Size: 2 * num_eras * B (one for grad, one for hess)
    # Assuming num_eras and B are small enough to fit in shared mem (<48KB)
    shared_grad = tl.zeros([ERA_BIN_SIZE], dtype=tl.float32)
    shared_hess = tl.zeros([ERA_BIN_SIZE], dtype=tl.float32)
    
    # Initialize shared memory to zero (distributed across threads)
    thread_idx = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, ERA_BIN_SIZE, BLOCK_SIZE, num_stages=1):
        idx = i + thread_idx
        mask_init = idx < ERA_BIN_SIZE
        tl.store(shared_grad + idx, 0.0, mask=mask_init)
        tl.store(shared_hess + idx, 0.0, mask=mask_init)
    
    tl.sync_barrier()  # Ensure all threads have initialized shared mem
    
    # Local aggregation: Atomic add to shared memory (faster than global)
    # Offset in shared: era * B + bin
    local_offset = eras * B + bins
    tl.atomic_add(shared_grad + local_offset, residuals, mask=row_mask)
    tl.atomic_add(shared_hess + local_offset, 1.0, mask=row_mask)
    
    tl.sync_barrier()  # Sync after all local additions
    
    # Reduction: Aggregate from shared to global memory
    # Distribute the work: each thread handles a portion of the era-bin space
    # Global offset: era * (F_active * B) + feat_order_idx * B + bin
    for i in tl.range(0, ERA_BIN_SIZE, BLOCK_SIZE, num_stages=1):
        idx = i + thread_idx
        mask_reduce = idx < ERA_BIN_SIZE
        era = idx // B
        bin_val = idx % B
        
        # Load from shared
        grad_val = tl.load(shared_grad + idx, mask=mask_reduce, other=0.0)
        hess_val = tl.load(shared_hess + idx, mask=mask_reduce, other=0.0)
        
        # Compute global offset
        global_offset = era * (F_active * B) + feat_order_idx * B + bin_val
        
        # Atomic add to global
        tl.atomic_add(grad_hist_ptr + global_offset, grad_val, mask=mask_reduce)
        tl.atomic_add(hess_hist_ptr + global_offset, hess_val, mask=mask_reduce)


# @triton.jit
# def _histogram_kernel(
#     bin_ptr,            # Pointer to bin indices [N, F_master]
#     res_ptr,            # Pointer to residuals [N]
#     sample_idx_ptr,     # Pointer to active sample indices [N_active]
#     feat_idx_ptr,       # Pointer to active feature mapping [F_active]
#     era_idx_ptr,        # Pointer to era indices [N]
#     grad_hist_ptr,      # Output: Gradient histogram [num_eras, F_active, B]
#     hess_hist_ptr,      # Output: Hessian histogram [num_eras, F_active, B]
#     N,                  # Total number of active samples
#     F_master,           # Total features in the original dataset
#     F_active,           # Number of features we are currently processing
#     B,                  # Number of bins (e.g., 5)
#     num_eras,           # Number of eras
#     BLOCK_SIZE: tl.constexpr
# ):
#     # 1. Thread Mapping
#     # Program ID 0 manages which feature we are processing
#     # Program ID 1 manages the tile/block of samples
#     feat_order_idx = tl.program_id(0) 
#     tile_idx = tl.program_id(1)      
    
#     # Load the actual feature index from the mapping table
#     feat_idx = tl.load(feat_idx_ptr + feat_order_idx)
    
#     # Calculate sample offsets for this block
#     row_offsets = tile_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     row_mask = row_offsets < N
    
#     # 2. Data Loading (Coalesced Memory Access)
#     # Load indices of samples belonging to the current node/active set
#     s_indices = tl.load(sample_idx_ptr + row_offsets, mask=row_mask)
    
#     # Load era and bin assignments for these specific samples
#     # bins: find the bin for the specific sample and feature
#     eras = tl.load(era_idx_ptr + s_indices, mask=row_mask)
#     bins = tl.load(bin_ptr + s_indices * F_master + feat_idx, mask=row_mask)
#     residuals = tl.load(res_ptr + s_indices, mask=row_mask)

#     # 3. Destination Address Calculation
#     # The histogram is stored in a 3D layout: [era, feature, bin]
#     # We flatten this to a 1D offset: 
#     # offset = (era * total_features_in_output * bins_per_feat) + (feat_idx * bins_per_feat) + bin
#     output_offset = eras * (F_active * B) + feat_order_idx * B + bins
    
#     # 4. Atomic Aggregation
#     # Instead of looping through all possible bins and eras, we "scatter" the 
#     # values directly to their destination using hardware-accelerated atomics.
#     # At B=5, the L2 cache efficiently handles collisions.
#     tl.atomic_add(grad_hist_ptr + output_offset, residuals, mask=row_mask)
#     tl.atomic_add(hess_hist_ptr + output_offset, 1.0, mask=row_mask)



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
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # N to liczba próbek, T to liczba drzew
    # Poprawka maskowania, aby nie wyjść poza zakres
    mask = idx < (N * T)
    
    s_idx = idx % N  # Indeks próbki
    t_idx = idx // N # Indeks drzewa
    
    node_id = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    active = mask
    
    # Przechodzenie po drzewie
    
    for _ in range(64):
        # 1. Definiujemy bazę dla węzła (tylko dla tych, co są w grze)
        tree_node_base = t_idx * max_nodes * 6 + node_id * 6
        
        # 2. Sprawdzamy, czy to liść
        # Ustawiamy mask=active, aby nie czytać śmieci z pamięci
        is_leaf = tl.load(tree_ptr + tree_node_base + 4, mask=active, other=0).to(tl.float32)
        
        # 3. Jeśli to liść I wątek jest aktywny -> zapisujemy wynik i dezaktywujemy wątek
        leaf_mask = active & (is_leaf > 0.5)
        
        # Wartość liścia (indeks 5)
        leaf_val = tl.load(tree_ptr + tree_node_base + 5, mask=leaf_mask, other=0.0)
        tl.atomic_add(out_ptr + s_idx, leaf_val * lr, mask=leaf_mask)
        
        # Kluczowe: wątki, które trafiły na liść, przestają być aktywne w kolejnych iteracjach
        active = active & (~leaf_mask)
        
        # 4. Dla reszty aktywnych wątków: idziemy głębiej w drzewo
        # Używamy mask=active, aby uniknąć błędów Out of Bounds
        feat = tl.load(tree_ptr + tree_node_base + 0, mask=active, other=0).to(tl.int32)
        split_bin = tl.load(tree_ptr + tree_node_base + 1, mask=active, other=0)
        
        # Pobranie wartości cechy dla danej próbki
        bin_val = tl.load(bin_ptr + s_idx * F + feat, mask=active, other=0)
        
        left_id = tl.load(tree_ptr + tree_node_base + 2, mask=active, other=0).to(tl.int32)
        right_id = tl.load(tree_ptr + tree_node_base + 3, mask=active, other=0).to(tl.int32)
        
        # Aktualizacja node_id dla następnej iteracji
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
    
    # Clear histograms before use (on CPU, or move to kernel if needed)
    grad_hist.fill_(0)
    hess_hist.fill_(0)
    
    # Precompute ERA_BIN_SIZE as constexpr
    ERA_BIN_SIZE = num_eras * num_bins
    
    # Ensure it fits in shared mem (add check if needed)
    assert ERA_BIN_SIZE * 4 * 2 < 48000, "Shared memory overflow: reduce num_eras or B"
    
    # Grid computation: Use meta['BLOCK_SIZE'] in lambda for autotune compatibility
    grid = lambda meta: (F_active, triton.cdiv(N, meta['BLOCK_SIZE']))
    
    _histogram_kernel[grid](
        bin_indices, residuals, sample_indices, feature_indices, era_indices,
        grad_hist, hess_hist, N, F_master, F_active, num_bins, num_eras,
        ERA_BIN_SIZE=ERA_BIN_SIZE  # Pass as constexpr
    )


#OLD
# def compute_histogram3(bin_indices, residuals, sample_indices, feature_indices, era_indices,
#                        grad_hist, hess_hist, num_bins, threads_per_block=256, rows_per_thread=1):
#     N = sample_indices.size(0)
#     F_active = feature_indices.size(0)
#     F_master = bin_indices.size(1)
#     num_eras = grad_hist.size(0)
    
#     # Clear histograms before use (on CPU, or move to kernel if needed)
#     grad_hist.fill_(0)
#     hess_hist.fill_(0)
    
#     # Precompute ERA_BIN_SIZE as constexpr
#     ERA_BIN_SIZE = num_eras * num_bins
    
#     # Ensure it fits in shared mem (add check if needed)
#     assert ERA_BIN_SIZE * 4 * 2 < 48000, "Shared memory overflow: reduce num_eras or B"
    
#     grid = (F_active, triton.cdiv(N, threads_per_block))
#     _histogram_kernel[grid](
#         bin_indices, residuals, sample_indices, feature_indices, era_indices,
#         grad_hist, hess_hist, N, F_master, F_active, num_bins, num_eras,
#         BLOCK_SIZE=threads_per_block,
#         ERA_BIN_SIZE=ERA_BIN_SIZE  # Pass as constexpr
#     )
    
# def compute_histogram3(bin_indices, residuals, sample_indices, feature_indices, era_indices, 
#                        grad_hist, hess_hist, num_bins, threads_per_block=256, rows_per_thread=1):
#     N = sample_indices.size(0)
#     F_active = feature_indices.size(0)
#     F_master = bin_indices.size(1)
#     num_eras = grad_hist.size(0)
    
#     # Czyścimy histogramy przed użyciem
#     grad_hist.fill_(0)
#     hess_hist.fill_(0)

#     grid = (F_active, triton.cdiv(N, threads_per_block))
#     _histogram_kernel[grid](
#         bin_indices, residuals, sample_indices, feature_indices, era_indices,
#         grad_hist, hess_hist, N, F_master, F_active, num_bins, num_eras,
#         BLOCK_SIZE=threads_per_block
#     )

def compute_split(G, H, min_split_gain, min_child_samples, eps, per_era_gain, per_era_direction, threads=128):
    E, F, B = G.shape
    grid = (triton.cdiv(F, threads), E)
    _split_kernel[grid](
        G, H, per_era_gain, per_era_direction, E, F, B,
        min_child_samples, eps, BLOCK_SIZE=threads
    )

def predict_forest(bin_indices, tree_tensor, learning_rate, out, block_size=512):
    N, F = bin_indices.shape
    T, max_nodes, _ = tree_tensor.shape
    grid = lambda meta: (triton.cdiv(N * T, meta['BLOCK_SIZE']),)
    _predict_kernel[grid](
        bin_indices, tree_tensor, out, N, F, T, max_nodes, learning_rate, BLOCK_SIZE=block_size
    )