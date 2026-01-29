from cs336_basics import model
import torch
from einops import einsum, rearrange
import math
import triton
import triton.language as tl

def make_attn_inputs(device=None):
    torch.random.manual_seed(0)
    batch_size = 4
    n_queries = 128
    n_keys = 128
    D = 64
    q = torch.randn(batch_size, n_queries, D, device=device, requires_grad=True)
    k = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
    v = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
    do = torch.randn(batch_size, n_queries, D, device=device)
    return q, k, v, do

def split_into_tiles(tensor, tile_size) -> torch.Tensor:
    return rearrange(tensor, "... (num_tiles tile_size) d -> ... num_tiles tile_size d", tile_size=tile_size)

def combine_tiles(tensor) -> torch.Tensor:
    return rearrange(tensor,'... num_tiles tile_size d -> ... (num_tiles tile_size) d')

class flash_attention_pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, tile_size=16):
        batch_size = Q.shape[0]
        seq_length = Q.shape[1]
        d_head = Q.shape[2]

        O = torch.full([batch_size, seq_length, d_head], 0.0, device=Q.device)
        L = torch.full([batch_size, seq_length], 0.0, device=Q.device)

        Q_tiles = split_into_tiles(Q, tile_size) # shape[batch_size, num_tiles, tile_size, d_head]
        K_tiles = split_into_tiles(K, tile_size) # shape[batch_size, num_tiles, tile_size, d_head]
        V_tiles = split_into_tiles(V, tile_size) # shape[batch_size, num_tiles, tile_size, d_head]
        for batch in range(len(Q_tiles)):
            for i in range(len(Q_tiles[batch])):
                O_ij = torch.full([tile_size, d_head], 0.0, device=Q.device)
                l_ij = torch.full([tile_size,], 0.0, device=Q.device)
                m_ij = torch.full([tile_size,], -torch.inf, device=Q.device)

                # Load Q_i from global memory
                Q_i:torch.Tensor = Q_tiles[batch][i]
                for j in range(len(K_tiles[batch])):
                    # Load K_j and V_j from global memory
                    K_j:torch.Tensor = K_tiles[batch][j]
                    V_j:torch.Tensor = V_tiles[batch][j]

                    # Compute tile of pre-softmax attention scores
                    S_ij = einsum(Q_i, K_j, "... q_tile_size d, ... k_tile_size d -> q_tile_size k_tile_size")/math.sqrt(d_head)

                    # Compute and updata maximum value
                    row_max, _ = torch.max(S_ij, dim=1)
                    m_ij_step_before = m_ij
                    m_ij = torch.maximum(m_ij, row_max)
                    
                    # Compute proxy
                    S_subtract_maximum = S_ij - m_ij.unsqueeze(-1)
                    P_ij = torch.exp(S_subtract_maximum)

                    # Compute l_ij
                    l_ij = torch.exp(m_ij_step_before - m_ij)*l_ij + torch.sum(P_ij, dim=-1)

                    # Compute O_ij
                    # O_ij = einsum(torch.diag(torch.exp(m_ij_step_before - m_ij)), O_ij, "B_q B_q, B_q d -> B_q d") + einsum(P_ij, V_j, "B_q B_k, B_k d -> B_q d")
                    O_ij = torch.exp(m_ij_step_before - m_ij)[:,None] * O_ij + einsum(P_ij, V_j, "q_tile_size k_tile_size, k_tile_size d -> q_tile_size d")

                # Compute O_i in total
                # O_ij = einsum(torch.inverse((torch.diag(l_ij))), O_ij, "tile_size B_k, B_k d -> tile_size d")
                O_ij = O_ij / l_ij[:, None]

                # Compute L_i 
                L_i = m_ij + torch.log(l_ij)

                # Write O_i to global memory as the i-th tile of O
                O[batch, i*tile_size:(i+1)*tile_size] = O_ij

                # Write L_i to global memory as the i-th tile of L
                L[batch, i*tile_size:(i+1)*tile_size] = L_i
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.tile_size = tile_size
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        tile_size = ctx.tile_size
        is_causal = ctx.is_causal
        d_head = Q.shape[2]

        Q_tiles = split_into_tiles(Q, tile_size) # shape[batch_size, num_tiles, Q_tile_size, d_head]
        O_tiles = split_into_tiles(O, tile_size) # shape[batch_size, num_tiles, Q_tile_size, d_head]
        dO_tiles = split_into_tiles(dO, tile_size) # shape[batch_size, num_tiles, Q_tile_size, d_head]

        K_tiles = split_into_tiles(K, tile_size) # shape[batch_size, num_tiles, tile_size, d_head]
        V_tiles = split_into_tiles(V, tile_size) # shape[batch_size, num_tiles, tile_size, d_head]

        L_tiles = rearrange(L, "batch (num_tiles tile_size) -> batch num_tiles tile_size", tile_size=tile_size) # shape[batch_size, num_tiles, tile_size]
        DEVICE = Q.device
        # Ouput gradients
        dQ = torch.zeros_like(Q,dtype=Q.dtype, device=DEVICE)
        dK = torch.zeros_like(K,dtype=K.dtype, device=DEVICE)
        dV = torch.zeros_like(V,dtype=V.dtype, device=DEVICE)

        dQ_tiles = split_into_tiles(dQ, tile_size)

        for batch in range(len(Q_tiles)):
            for j in range(len(K_tiles[batch])):
                # Load K_j V_j from global memory
                K_j = K_tiles[batch, j] # shape = [k_tile_size, d]
                V_j = V_tiles[batch, j] # shape = [k_tile_size, d]

                # Initialize dK_j = dV_j = 0 shape = [K_tile_size, d]
                dK_j = torch.zeros_like(K_j, device=DEVICE)
                dV_j = torch.zeros_like(V_j, device=DEVICE)

                if is_causal:
                    k_idx = torch.arange(j * tile_size, (j + 1) * tile_size, device=DEVICE)

                for i in range(len(Q_tiles[batch])):
                    # Load Q_i, O_i, dO_i, dQ_i, L_i from global memory
                    Q_i = Q_tiles[batch, i]
                    O_i = O_tiles[batch, i] # shape = [Q_tile_size, d]
                    dO_i = dO_tiles[batch, i] # shape [Q_tile_size, d]
                    L_i = L_tiles[batch, i]

                    # Compute tile of attention scores S_ij
                    S_ij = (einsum(Q_i, K_j, 'Q_tile_size d, K_tile_size d -> Q_tile_size K_tile_size') / math.sqrt(d_head)) # shape(q_tile_size, k_tile_size)

                    # Causal mask (if enabled), mask j>i entries
                    if is_causal:
                        q_idx = torch.arange(i * tile_size, (i + 1) * tile_size, device=DEVICE)
                        causal = k_idx[None, :] > q_idx[:, None]  # [Bq, Bk]
                        S_ij = S_ij.masked_fill(causal, -1e6)

                    # Compute attention probabilities P_ij
                    P_ij = torch.exp(S_ij - L_i[:, None]) # shape (q_tile_size k_tile_size)

                    # Compute and accumulate dV_j
                    dV_j += einsum(P_ij, dO_i, 'q_tile_size k_tile_size, q_tile_size d_head -> k_tile_size d_head') # shape (k_tile_size d_head)

                    # Compute dP_ij
                    dP_ij = einsum(dO_i, V_j,'Q_tile_size d, K_tile_size d ->Q_tile_size K_tile_size')

                    # Compute dS_ij
                    D_i = torch.sum(dO_i * O_i, dim=-1)[:, None] # shape(q_tile_size, 1)
                    dS_ij = P_ij * (dP_ij - D_i) / math.sqrt(d_head) # shape (Q_tile_size, K_tile_size)

                    # Load dQ_i from global memroy then update  dQ_i += dS_ij K_j shape=(Q_tile_size, d_head) then write it back to global memroy
                    dQ_i = dQ_tiles[batch, i]
                    dQ_i += einsum(dS_ij, K_j,'Q_tile_size K_tile_size, K_tile_size d_head -> Q_tile_size d_head') # shape (Q_tile_size, d_head)

                    # Compute and accumulate dK_j
                    dK_j += einsum(dS_ij, Q_i, 'Q_tile_size K_tile_size, Q_tile_size d_head -> K_tile_size d_head')

                # Write dK_j and dV_j to global memory as the j-th tiles of dK and DV
                dK[batch, j*tile_size:(j+1)*tile_size] = dK_j
                dV[batch, j*tile_size:(j+1)*tile_size] = dV_j

        return dQ, dK, dV, None, None

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero') # (Q_TILE_SIZE, D)

    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)

    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero') # (K_TILE_SIZE, D)
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero') # (K_TILE_SIZE, D)
        
        # Compute tile of pre-softmax attention scores make S_P save scores
        S_P = tl.dot(Q,tl.trans(K,(1, 0))) * scale # shape (Q_TILE_SIZE, K_TILE_SIZE)
        S_P = tl.cast(S_P, tl.float32)
        if (is_causal):
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = k_idx[None, :] > q_idx[:, None]  # shape (Q_TILE_SIZE, K_TILE_SIZE)
            S_P = S_P + mask * (-1e6)

        # Compute and updata maximum value
        row_max = tl.max(S_P, axis=1)
        m_step_before = m
        m = tl.maximum(m, row_max) # shape (Q_TILE_SIZE)

        # Compute proxy change S_P to save proxy
        S_P = S_P - m[:, None]
        S_P = tl.exp(S_P) # shape (Q_TILE_SIZE, K_TILE_SIZE)

        # Compute l
        l = tl.exp(m_step_before - m)*l + tl.sum(S_P, axis=1) # shape (Q_TILE_SIZE)

        # Compute output
        output = tl.exp(m_step_before - m)[:, None] * output + tl.dot(S_P, tl.cast(V, tl.float32))      # (Q_TILE_SIZE, D)

        # Move into next tile for K, V
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Compute output in total
    output = output / l[:, None]

    # Compute logsumexp 
    l = m + tl.log(l)

    # Write output and logsumexp to global memory
    tl.store(O_block_ptr, output, boundary_check=(1, 0))
    tl.store(L_block_ptr, l, boundary_check=(0,))

@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, dO_ptr, # Input pointer
    dK_ptr, dV_ptr, # Output pointer
    stride_qb, stride_qq, stride_qd, # Q.shape [batch_size, n_queries, d]
    stride_kb, stride_kk, stride_kd, # K.shape [batch_size, n_keys, d]
    stride_vb, stride_vk, stride_vd, # V.shape [batch_size, n_keys, d]
    stride_ob, stride_oq, stride_od, # O.shape [batch_size, n_queries, d]
    stride_lb, stride_lq, # L.shape [batch_size, n_queries]
    stride_dob, stride_doq, stride_dod, # dO.shape [batch_size, n_queries, d]
    stride_dkb, stride_dkk, stride_dkd, # dK.shape [batch_size, n_keys, d]
    stride_dvb, stride_dvk, stride_dvd, # dV.shape [batch_size, n_keys, d]
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Program indices
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0, ),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Load K, V from global memory
    K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero') # (K_TILE_SIZE, D)
    V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero') # (K_TILE_SIZE, D)

    # Initiliaze dK = dV = 0 get shape (K_TILE_SIZE, D)
    dK = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    
    for query_tile_index in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        # Load Q, O, dO from global memory
        Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero') # (Q_TILE_SIZE, D)
        O = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option='zero') # (Q_TILE_SIZE, D)
        dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option='zero') # (Q_TILE_SIZE, D)
        L = tl.load(L_block_ptr, boundary_check=(0,), padding_option='zero') # (Q_TILE_SIZE,)

        # Compute tile of attention scores S
        S = tl.dot(Q, tl.trans(K, (1, 0))) * scale
        S = tl.cast(S, tl.float32) # for exp/softmax math
        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = k_idx[None, :] > q_idx[:, None]
            S = S + mask * (-1e6)

        # Compute attention probabilities P
        P = tl.exp(S - tl.cast(L[:, None], tl.float32)) # shape (Q_TILE_SIZE, K_TILE_SIZE)
        P = tl.cast(P ,Q.dtype)

        # Compute and accumulate dV
        dO = tl.cast(dO, Q.dtype)
        dV += tl.dot(tl.trans(P, (1, 0)), dO) # shape (K_TILE_SIZE, D)

        # Compute dP
        dP = tl.dot(dO, tl.trans(V, (1, 0))) # shape (Q_TILE_SIZE, K_TILE_SIZE)
        dP = tl.cast(dP, tl.float32)

        # Compute dS
        # dS = (P * (dP - (tl.sum((dO * O), axis=1)[:, None]))) / (D ** 0.5)
        Dvec = tl.sum(tl.cast(dO, tl.float32) * tl.cast(O, tl.float32), axis=1)[:, None]  # (Bq,1)
        dS = P * (dP - Dvec) * scale
        dS = tl.cast(dS, Q.dtype)

        # Load dQ from global memroy then update  dQ += dS K shape=(Q_tile_size, d_head) then write it back to global memroy
        # Has been removed since we create a new kernel to aviod atomic read and write
        
        # Compute and accumulate dK
        dK += tl.dot(tl.trans(dS,(1, 0)), Q)

        # Move pointer to next step
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
    
    # Writee dK and dV back to global memory
    tl.store(dK_block_ptr, dK, boundary_check=(1, 0))
    tl.store(dV_block_ptr, dV, boundary_check=(1, 0))
    
@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, dO_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    q_tile = tl.program_id(0)
    b = tl.program_id(1)

    # Block pointers for Q, O, dO, L for THIS q_tile
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + b * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + b * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_tile * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + b * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(q_tile * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + b * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_tile * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    O = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    L = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")  # logsumexp per row

    # Dvec = sum(dO * O) per query row (compute once per q_tile)
    Dvec = tl.sum(tl.cast(dO, tl.float32) * tl.cast(O, tl.float32), axis=1)[:, None]  # (Q_TILE_SIZE, 1)

    dQ_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # Loop over key tiles
    K_block_ptr = tl.make_block_ptr(
        K_ptr + b * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + b * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    for k_tile in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D)
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D)

        S = tl.dot(Q, tl.trans(K, (1, 0))) * scale
        S = tl.cast(S, tl.float32)

        if is_causal:
            q_rows = q_tile * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]
            k_cols = k_tile * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :]
            S = tl.where(q_rows >= k_cols, S, -1e6)

        P = tl.exp(S - tl.cast(L[:, None], tl.float32))  # (Q_TILE_SIZE, K_TILE_SIZE)

        dP = tl.dot(tl.cast(dO, Q.dtype), tl.trans(tl.cast(V, Q.dtype), (1, 0)))
        dP = tl.cast(dP, tl.float32)

        dS = P * (dP - Dvec) * scale
        dS = tl.cast(dS, Q.dtype)

        dQ_acc += tl.dot(dS, K)  # accum fp32

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Store dQ (no atomics!)
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + b * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(q_tile * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    tl.store(dQ_block_ptr, dQ_acc, boundary_check=(0, 1))


class flash_attention_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, is_causal=False, tile_size=16):
        orig_shape = Q.shape
        Q_ = Q.reshape(-1, Q.shape[-2], Q.shape[-1])
        K_ = K.reshape(-1, K.shape[-2], K.shape[-1])
        V_ = V.reshape(-1, V.shape[-2], V.shape[-1])

        BATCH_SIZE = Q_.shape[0]
        N_QUERIES = Q_.shape[1]
        N_KEYS = K_.shape[1]
        D = Q_.shape[2]
        scale = 1/math.sqrt(D)
        Q_TILE_SIZE = tile_size
        K_TILE_SIZE = tile_size
        Tq = triton.cdiv(N_QUERIES, Q_TILE_SIZE)  # number of query tiles

        Output = torch.zeros([BATCH_SIZE, N_QUERIES, D], device=Q.device)
        Logsumexp = torch.zeros([BATCH_SIZE, N_QUERIES], device=Q.device)

        grid = (Tq, BATCH_SIZE)
        flash_fwd_kernel[grid](
            Q_ptr=Q_, K_ptr=K_, V_ptr=V_,
            O_ptr=Output, L_ptr=Logsumexp,
            stride_qb=(N_QUERIES*D), stride_qq=(D), stride_qd=(1),
            stride_kb=(N_KEYS*D), stride_kk=(D), stride_kd=(1),
            stride_vb=(N_KEYS*D), stride_vk=(D), stride_vd=(1),
            stride_ob=(N_QUERIES*D), stride_oq=(D), stride_od=(1),
            stride_lb=(N_QUERIES), stride_lq=(1),
            N_QUERIES=N_QUERIES, N_KEYS=N_KEYS,
            scale=scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal
        )
        ctx.is_causal = is_causal
        ctx.save_for_backward(Q_, K_, V_, Output.to(Q.dtype), Logsumexp.to(Q.dtype))
        ctx.tile_size = tile_size
        ctx.orig_shape = orig_shape
        Output = Output.reshape(orig_shape)
        return Output.to(Q.dtype)
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        tile_size = ctx.tile_size
        orig_shape = ctx.orig_shape

        BATCH_SIZE = Q.shape[0]
        N_QUERIES = Q.shape[1]
        N_KEYS = K.shape[1]
        D = Q.shape[2]
        scale = 1/math.sqrt(D)
        Q_TILE_SIZE = tile_size
        K_TILE_SIZE = tile_size
        Tq = triton.cdiv(N_QUERIES, Q_TILE_SIZE)
        Tk = triton.cdiv(N_KEYS, K_TILE_SIZE)  # number of query tiles

        dQ = torch.zeros_like(Q, device=Q.device, dtype=torch.float32)
        dK = torch.zeros_like(K, device=K.device, dtype=torch.float32)
        dV = torch.zeros_like(V, device=V.device, dtype=torch.float32)

        dO_ = dO.reshape(-1, Q.shape[-2], Q.shape[-1])

        grid = (Tk, BATCH_SIZE)
        flash_bwd_dkdv_kernel[grid](
            Q_ptr=Q, K_ptr=K, V_ptr=V, O_ptr=O, L_ptr=L, dO_ptr=dO_, # Input pointer
            dK_ptr=dK, dV_ptr=dV, # Output pointer
            stride_qb=(N_QUERIES*D), stride_qq=(D), stride_qd=(1), # Q.shape [batch_size, n_queries, d]
            stride_kb=(N_KEYS*D), stride_kk=(D), stride_kd=(1), # K.shape [batch_size, n_keys, d]
            stride_vb=(N_KEYS*D), stride_vk=(D), stride_vd=(1), # V.shape [batch_size, n_keys, d]
            stride_ob=(N_QUERIES*D), stride_oq=(D), stride_od=(1), # O.shape [batch_size, n_queries, d]
            stride_lb=(N_QUERIES), stride_lq=(1), # L.shape [batch_size, n_queries]
            stride_dob=(N_QUERIES*D), stride_doq=(D), stride_dod=(1), # dO.shape [batch_size, n_queries, d]
            stride_dkb=(N_KEYS*D), stride_dkk=(D), stride_dkd=(1), # dK.shape [batch_size, n_keys, d]
            stride_dvb=(N_KEYS*D), stride_dvk=(D), stride_dvd=(1), # dV.shape [batch_size, n_keys, d]
            N_QUERIES=N_QUERIES, N_KEYS=N_KEYS,
            scale=scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal
        )

        # dQ kernel: grid is (q_tiles, batch)
        flash_bwd_dq_kernel[(Tq, BATCH_SIZE)](
            Q_ptr=Q, K_ptr=K, V_ptr=V, O_ptr=O, L_ptr=L, dO_ptr=dO_,
            dQ_ptr=dQ,
            stride_qb=(N_QUERIES*D), stride_qq=(D), stride_qd=(1),
            stride_kb=(N_KEYS*D), stride_kk=(D), stride_kd=(1),
            stride_vb=(N_KEYS*D), stride_vk=(D), stride_vd=(1),
            stride_ob=(N_QUERIES*D), stride_oq=(D), stride_od=(1),
            stride_lb=(N_QUERIES), stride_lq=(1),
            stride_dob=(N_QUERIES*D), stride_doq=(D), stride_dod=(1),
            stride_dqb=(N_QUERIES*D), stride_dqq=(D), stride_dqd=(1),
            N_QUERIES=N_QUERIES, N_KEYS=N_KEYS,
            scale=scale,
            D=D, Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        dQ = dQ.reshape(orig_shape)
        dK = dK.reshape(orig_shape)
        dV = dV.reshape(orig_shape)
        return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype), None, None