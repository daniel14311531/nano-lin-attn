import pytest
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, sys.path[0] + '/../../..')

from ops.o2b_deltanet import o2b_delta_rule, o2b_delta_rule_recurrent, O2B_state

# dtype = torch.float32
dtype = torch.bfloat16

# Adjust tolerances based on dtype precision
# bfloat16 has ~7-8 bits of mantissa (vs 23 for float32), so we need looser tolerances
# Forward pass can achieve ~1e-2, backward pass needs ~1e-1 for bfloat16
if dtype == torch.bfloat16:
    TOLERANCE_ATOL_FWD = 1e-2
    TOLERANCE_RTOL_FWD = 1e-2
    TOLERANCE_ATOL_BWD = 3e-1  # Looser for backward pass
    TOLERANCE_RTOL_BWD = 3e-1
else:
    TOLERANCE_ATOL_FWD = 1e-3
    TOLERANCE_RTOL_FWD = 1e-3
    TOLERANCE_ATOL_BWD = 1e-3
    TOLERANCE_RTOL_BWD = 1e-3

def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute relative error between two tensors."""
    diff = (a - b).abs()
    scale = a.abs() + b.abs()
    # Avoid division by zero
    scale = torch.clamp(scale, min=1e-8)
    return (diff / scale).max().item()


def print_error_stats(name: str, a: torch.Tensor, b: torch.Tensor):
    """Print error statistics for two tensors."""
    rel_err = relative_error(a, b)
    abs_err = (a - b).abs().max().item()
    print(f"  {name:12s}: rel_err={rel_err:.6e}, abs_err={abs_err:.6e}")


@pytest.mark.parametrize("B,L,H,D", [(2, 128, 4, 8), (4, 200, 2, 16)])
def test_o2b_delta_rule_forward(B, L, H, D):
    """Test forward pass matches recurrent implementation."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    k = F.normalize(torch.randn(B, L, H, D, device=device, dtype=dtype), dim=-1)
    q = F.normalize(torch.randn(B, L, H, D, device=device, dtype=dtype), dim=-1)
    v = torch.randn(B, L, H, D, device=device, dtype=dtype)
    b = torch.randn(B, L, H, D, device=device, dtype=dtype) * 0.1

    init_state: O2B_state = (
        torch.zeros(B, H, D, D, device=device, dtype=dtype),
        torch.zeros(B, H, D, D, device=device, dtype=dtype),
        torch.tensor(1, device=device),
    )

    o_parallel, state_parallel = o2b_delta_rule(k, q, v, b, init_state)
    o_recurrent, state_recurrent = o2b_delta_rule_recurrent(k, q, v, b, init_state)

    print(f"\n[B={B}, L={L}, H={H}, D={D}] Forward pass error stats:")
    print_error_stats("output", o_parallel, o_recurrent)
    print_error_stats("W_t", state_parallel[0], state_recurrent[0])
    print_error_stats("W_avg", state_parallel[1], state_recurrent[1])

    assert torch.allclose(o_parallel, o_recurrent, atol=TOLERANCE_ATOL_FWD, rtol=TOLERANCE_RTOL_FWD), "Forward output mismatch"
    assert torch.allclose(state_parallel[0], state_recurrent[0], atol=TOLERANCE_ATOL_FWD, rtol=TOLERANCE_RTOL_FWD), "W_t mismatch"
    assert torch.allclose(state_parallel[1], state_recurrent[1], atol=TOLERANCE_ATOL_FWD, rtol=TOLERANCE_RTOL_FWD), "W_avg mismatch"
    assert state_parallel[2] == state_recurrent[2], "t mismatch"


@pytest.mark.parametrize("B,L,H,D", [(2, 128, 4, 8), (4, 200, 2, 16)])
def test_o2b_delta_rule_backward(B, L, H, D):
    """Test backward pass gradients match recurrent implementation."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Shared random seed for reproducibility
    torch.manual_seed(42)

    def create_inputs():
        k_raw = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
        k = F.normalize(k_raw, dim=-1)
        k.retain_grad()

        q_raw = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
        q = F.normalize(q_raw, dim=-1)
        q.retain_grad()

        v = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)

        b_raw = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
        b = b_raw * 0.1
        b.retain_grad()

        return k, q, v, b

    # Create identical inputs for both implementations
    k1, q1, v1, b1 = create_inputs()
    torch.manual_seed(42)  # Reset seed for identical inputs
    k2, q2, v2, b2 = create_inputs()

    init_state: O2B_state = (
        torch.zeros(B, H, D, D, device=device, dtype=dtype),
        torch.zeros(B, H, D, D, device=device, dtype=dtype),
        torch.tensor(1, device=device),
    )

    o_parallel, _ = o2b_delta_rule(k1, q1, v1, b1, init_state)
    o_recurrent, _ = o2b_delta_rule_recurrent(k2, q2, v2, b2, init_state)

    # Same grad output for both
    grad_output = torch.randn_like(o_parallel)
    o_parallel.backward(grad_output)
    o_recurrent.backward(grad_output.clone())

    print(f"\n[B={B}, L={L}, H={H}, D={D}] Backward pass gradient error stats:")
    print_error_stats("k.grad", k1.grad, k2.grad)
    print_error_stats("q.grad", q1.grad, q2.grad)
    print_error_stats("v.grad", v1.grad, v2.grad)
    print_error_stats("b.grad", b1.grad, b2.grad)

    assert torch.allclose(k1.grad, k2.grad, atol=TOLERANCE_ATOL_BWD, rtol=TOLERANCE_RTOL_BWD), "k grad mismatch"
    assert torch.allclose(q1.grad, q2.grad, atol=TOLERANCE_ATOL_BWD, rtol=TOLERANCE_RTOL_BWD), "q grad mismatch"
    assert torch.allclose(v1.grad, v2.grad, atol=TOLERANCE_ATOL_BWD, rtol=TOLERANCE_RTOL_BWD), "v grad mismatch"
    assert torch.allclose(b1.grad, b2.grad, atol=TOLERANCE_ATOL_BWD*2, rtol=TOLERANCE_RTOL_BWD*2), "b grad mismatch"  # Slightly looser for b
