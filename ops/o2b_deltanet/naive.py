import torch

O2B_state = tuple[torch.Tensor, torch.Tensor, torch.long]

CHUNK_SIZE = 64


def calc_inv(T: torch.Tensor):
    """
    Calculate the combined inverse of a strictly lower triangular tensor.

    Args:
        T: Input tensor of shape (..., C, C), with strictly lower triangular structure

    Returns:
        Inverse of (I + T)^{-1}
    """
    # B, H, C, _ = T.size()
    # dtype = T.dtype
    # res = T + torch.eye(C, device=T.device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # (B, H, C, C)
    # return torch.linalg.inv(res.float()).to(dtype)
    C = T.shape[-1]
    I = torch.eye(C, device=T.device, dtype=T.dtype).unsqueeze(0).unsqueeze(0)
    res = I
    mu = -T
    base = I
    t = C
    while t > 0:
        res = mu @ res + res
        mu = mu @ mu
        t /= 2
    return res


def o2b_delta_rule_recurrent(
    k: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    init_state: O2B_state,
):
    """
    Multi-step recurrent version for inference.

    Args:
        k: Key tensor of shape (B, L, H, D)
        q: Query tensor of shape (B, L, H, D)
        v: Value tensor of shape (B, L, H, D)
        b: combined tensor of shape (B, L, H, D)
        init_state: (W_t, W_avg, t) where
            W_t: current weight matrix (B, H, D, D)
            W_avg: running average of W_t (B, H, D, D)
            t: current step count

    Returns:
        o: Output tensor of shape (B, L, H, D)
        new_state: Updated (W_t, W_avg, t)
    """
    B, L, H, D = k.shape
    W_t, W_avg, t = init_state
    output_dtype = k.dtype

    # Convert state to float32 for precision if input is bfloat16
    use_float32_state = output_dtype == torch.bfloat16
    if use_float32_state:
        W_t = W_t.float()
        W_avg = W_avg.float()

    outputs = []

    # Process each timestep sequentially
    for i in range(L):
        k_t = k[:, i:i+1, :, :].transpose(1, 2)  # (B, H, 1, D)
        q_t = q[:, i:i+1, :, :].transpose(1, 2)  # (B, H, 1, D)
        v_t = v[:, i:i+1, :, :].transpose(1, 2)  # (B, H, 1, D)
        b_t = b[:, i:i+1, :, :].transpose(1, 2)  # (B, H, 1, D)

        # Update W_t: W_{t+1} = W_t + k @ (v^T - b^T @ W_t)
        # where b = eta * beta * k is the effective key
        e = v_t.float() - b_t.float() @ W_t
        W_t = W_t + k_t.transpose(-2, -1).float() @ e

        # Update running average: W_avg_{t+1} = (t * W_avg_t + W_t_new) / (t + 1)
        t_f = t.float() if isinstance(t, torch.Tensor) else float(t)
        W_avg = (t_f / (t_f + 1.0)) * W_avg + W_t / (t_f + 1.0)
        t = t + 1

        # Compute output: o = q @ W_avg
        o_t = q_t.float() @ W_avg
        if use_float32_state:
            o_t = o_t.to(output_dtype)
        outputs.append(o_t)

    o = torch.cat(outputs, dim=2).transpose(1, 2)  # (B, H, L, D) -> (B, L, H, D)

    # Convert state back to original dtype
    if use_float32_state:
        W_t = W_t.to(output_dtype)
        W_avg = W_avg.to(output_dtype)

    new_state = (W_t, W_avg, t)

    return o, new_state


def o2b_delta_rule(
    k: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    init_state: O2B_state,
):
    """
    Chunkwise parallel version of online-to-batch delta rule.

    Args:
        k: Key tensor of shape (B, L, H, D)
        q: Query tensor of shape (B, L, H, D)
        v: Value tensor of shape (B, L, H, D)
        b: combined tensor of shape (B, L, H, D)
        init_state: (W_t, W_avg, t) where
            W_t: current weight matrix (B, H, D, D)
            W_avg: running average of W_t (B, H, D, D)
            t: current step count

    Returns:
        Output tensor of shape (B, L, H, D)
        new_state: Updated (W_t, W_avg, t)
    """
    B, L, H, D = k.shape
    W_t, W_avg, t = init_state
    C = CHUNK_SIZE
    num_chunks = (L + C - 1) // C
    output_dtype = k.dtype

    # Convert state to float32 for precision if input is bfloat16
    use_float32_state = output_dtype == torch.bfloat16
    if use_float32_state:
        W_t = W_t.float()
        W_avg = W_avg.float()

    outputs = []

    for i in range(num_chunks):
        start = i * C
        end = min(start + C, L)
        actual_C = end - start

        # Get chunk data: (B, C, H, D) -> (B, H, C, D)
        K_i = k[:, start:end].transpose(1, 2)
        Q_i = q[:, start:end].transpose(1, 2)
        V_i = v[:, start:end].transpose(1, 2)
        B_i = b[:, start:end].transpose(1, 2)

        # U[i] = (I + tril(B[i]K[i]^T, -1))^{-1} (V[i] - B[i]W[i-1])
        T = torch.tril(B_i.float() @ K_i.transpose(-2, -1).float(), diagonal=-1)
        # inv = torch.linalg.inv(T + torch.eye(T.shape[-1], device=T.device, dtype=T.dtype).unsqueeze(0).unsqueeze(0))
        inv = calc_inv(T)
        U_i = inv @ (V_i.float() - B_i.float() @ W_t)

        # W[i] = W[i-1] + K[i]^T U[i]
        W_new = W_t + K_i.transpose(-2, -1).float() @ U_i

        # Compute W_avg using matrix form (efficient parallel computation)
        # Formula: W_avg_new = t/(t+C) * W_avg + C/(t+C) * W_t + (K_i^T U_i) @ diag(w_i)
        # where w_i[j] = (C-j) / (t+C), j = 0..C-1 (decreasing weights)
        t_start = t + i * C
        t_start_f = t_start.float() if t_start.is_floating_point() else float(t_start)
        actual_C_f = float(actual_C)
        idx = torch.arange(actual_C, device=k.device, dtype=torch.float32) + 1
        w_i = (actual_C_f + 1 - idx) / (t_start_f + actual_C_f)  # (C,), weights: [C/(t+C), (C-1)/(t+C), ..., 1/(t+C)]

        coef1 = t_start_f / (t_start_f + actual_C_f)
        coef2 = actual_C_f / (t_start_f + actual_C_f)
        W_avg_new = (coef1 * W_avg + coef2 * W_t +
                     (K_i.transpose(-2, -1).float() @ (U_i * w_i.view(1, actual_C, 1))))

        # a[i], c[i] for output weighting
        idx = torch.arange(actual_C, device=k.device, dtype=torch.float32) + 1
        a = t_start_f / (t_start_f + idx)
        c = 1 - a

        # T[i](r,j) = (r-j+1)/((i-1)C+r) if r>=j else 0
        r = torch.arange(actual_C, device=k.device, dtype=torch.float32).unsqueeze(1) + 1
        j_idx = torch.arange(actual_C, device=k.device, dtype=torch.float32).unsqueeze(0) + 1
        T_mat = torch.where(r >= j_idx, (r - j_idx + 1) / (t_start_f + r), torch.zeros_like(r))

        # O[i] = diag(a)Q[i]W_avg[i-1] + diag(c)Q[i]W[i-1] + (T[i] o (Q[i]K[i]^T))U[i]
        QW_avg = Q_i.float() @ W_avg
        QW_t = Q_i.float() @ W_t
        QK_T = Q_i.float() @ K_i.transpose(-2, -1).float()
        O_i_f32 = (QW_avg * a.view(1, actual_C, 1) +
                   QW_t * c.view(1, actual_C, 1) +
                   (T_mat * QK_T) @ U_i)

        if use_float32_state:
            O_i = O_i_f32.to(output_dtype)
        else:
            O_i = O_i_f32

        outputs.append(O_i.transpose(1, 2))
        W_t, W_avg = W_new, W_avg_new

    o = torch.cat(outputs, dim=1)

    # Convert state back to original dtype
    if use_float32_state:
        W_t = W_t.to(output_dtype)
        W_avg = W_avg.to(output_dtype)

    new_state = (W_t, W_avg, t + L)

    return o, new_state