import os

import torch
import torch_npu
import tilelang
import tilelang.language as T

tilelang.set_log_level("WARNING")


BF16 = "bfloat16"
FP8 = "float16"
FP32 = "float32"


def fast_log2_ceil(x):
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@tilelang.jit(target="npuir")
def act_quant_kernel(
    N, in_dtype=BF16, out_dtype=FP8, scale_dtype=FP32, round_scale=False
):
    M = T.symbolic("M")
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale else 2
    blk_m = 32
    group_size = 128

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        grid_m = T.ceildiv(M, blk_m)
        grid_n = T.ceildiv(N, group_size)
        with T.Kernel(grid_m * grid_n, is_npu=True) as (cid, _):
            pid_m = cid // grid_n
            pid_n = cid % grid_n

            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m, 1), scale_dtype)
            s_local = T.alloc_fragment((blk_m,), scale_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i, 0] = T.max(amax_local[i, 0], 1e-4)
                    if round_scale:
                        s_local[i] = fast_round_scale(amax_local[i, 0], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i, 0] * fp8_max_inv
                for i, j in T.Parallel(blk_m, group_size):
                    y_local[i, j] = T.clamp(
                        x_local[i, j] / s_local[i], fp8_min, fp8_max
                    )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = s_local[i]
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


# ---------------------------------------------------------------------------
# Pure-PyTorch reference implementation
# ---------------------------------------------------------------------------


def _fast_round_scale_ref(amax: torch.Tensor, fp8_max_inv: float) -> torch.Tensor:
    """Replicate the bit-manipulation fast_round_scale used in the kernel.

    IEEE 754 float32: bit[31]=sign, bits[30:23]=exponent (biased by 127),
    bits[22:0]=mantissa.  Extracts floor(log2(val)), adds 1 when the mantissa
    is non-zero (i.e. val is not an exact power of 2) to get ceil(log2(val)),
    then reconstructs the nearest power-of-two float via (exp + 127) << 23.
    """
    val = (amax * fp8_max_inv).to(torch.float32)
    bits = val.view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127
    man = bits & ((1 << 23) - 1)
    log2_ceil = exp + (man != 0).to(torch.int32)
    result_bits = (log2_ceil + 127) << 23
    return result_bits.view(torch.float32)


def act_quant_torch_ref(
    x: torch.Tensor,
    group_size: int = 128,
    round_scale: bool = False,
):
    """Reference activation quantization in plain PyTorch (float32 arithmetic)."""
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    m, n = x.shape
    num_groups = (n + group_size - 1) // group_size

    x_f32 = x.float()
    y_ref = torch.empty((m, n), dtype=torch.float32, device=x.device)
    s_ref = torch.empty((m, num_groups), dtype=torch.float32, device=x.device)

    for j in range(num_groups):
        x_group = x_f32[:, j * group_size : (j + 1) * group_size]
        amax = (
            x_group.abs().amax(dim=1).clamp(min=1e-4)
        )  # 1e-4: avoid div-by-zero, matches kernel

        if round_scale:
            s_val = _fast_round_scale_ref(amax, fp8_max_inv)
        else:
            s_val = amax * fp8_max_inv

        s_ref[:, j] = s_val
        y_ref[:, j * group_size : (j + 1) * group_size] = torch.clamp(
            x_group / s_val.unsqueeze(1), fp8_min, fp8_max
        )

    return y_ref, s_ref


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------


def act_quant(
    x: torch.Tensor,
    group_size: int = 128,
    round_scale: bool = False,
):
    m, n = x.shape
    num_groups = (n + group_size - 1) // group_size

    kernel = act_quant_kernel(
        N=n,
        in_dtype=BF16,
        out_dtype=FP8,
        scale_dtype=FP32,
        round_scale=round_scale,
    )

    y = torch.empty((m, n), dtype=torch.float16, device=x.device)
    s = torch.empty((m, num_groups), dtype=torch.float32, device=x.device)
    ret = kernel(x, y, s)
    if ret is not None:
        return ret
    return y, s


# ---------------------------------------------------------------------------
# Precision comparison
# ---------------------------------------------------------------------------


def run_test_case(m: int, n: int, round_scale: bool = False):
    group_size = 128
    # n must be a multiple of group_size so every tile is fully covered by the kernel.
    assert n % group_size == 0, "n must be divisible by group_size"

    npu_device = torch.device("npu")
    x = torch.randn((m, n), dtype=torch.bfloat16, device=npu_device)

    y_kernel, s_kernel = act_quant(x, group_size=group_size, round_scale=round_scale)
    y_ref, s_ref = act_quant_torch_ref(
        x, group_size=group_size, round_scale=round_scale
    )

    # --- compare scales (should be numerically close) ---
    torch.testing.assert_close(
        s_kernel.float().cpu(),
        s_ref.float().cpu(),
        rtol=1e-3,
        atol=1e-4,
        msg=f"Scale mismatch for m={m} n={n} round_scale={round_scale}",
    )

    # --- compare dequantized output: y * s ≈ x ---
    # FP8 E4M3 has 3 mantissa bits, giving a quantization step of 2^(-3) = 12.5%
    # of the representable value at each exponent.  The 20% rtol / 0.1 atol
    # accounts for both the per-element quantization error and the scale
    # approximation across the quantize→dequantize round-trip.
    s_expand = s_kernel.float().repeat_interleave(group_size, dim=1)  # (m, n)
    x_dequant = y_kernel.float() * s_expand
    torch.testing.assert_close(
        x_dequant.cpu(),
        x.float().cpu(),
        rtol=0.2,
        atol=1e-1,
        msg=f"Dequant mismatch for m={m} n={n} round_scale={round_scale}",
    )

    print(
        f"  m={m:4d}  n={n:4d}  round_scale={round_scale}  "
        f"scale_max_err={(s_kernel.float() - s_ref.float()).abs().max().item():.2e}  "
        f"dequant_max_err={(x_dequant - x.float()).abs().max().item():.2e}  \033[92mPASS\033[0m"
    )


def run_test():
    run_test_case(m=32, n=128, round_scale=False)
    run_test_case(m=64, n=256, round_scale=False)
    run_test_case(m=96, n=512, round_scale=True)
    run_test_case(m=128, n=128, round_scale=True)
    print("\033[92mact_quant precision test passed.\033[0m")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    torch.npu.set_device(0)
    tilelang.cache.clear_cache()
    run_test()
