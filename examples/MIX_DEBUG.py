# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import torch
import tilelang
import tilelang.language as T


seq_len = 512
dim = 16

torch.npu.set_device(11)


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def torch_online_flash_attention_blocked_debug(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_m: int,
    block_n: int,
    out_dtype: torch.dtype = torch.float16,
):
    """Blocked online flash-attention reference with intermediate DEBUG tensors.

    Returns:
        output: [seq_len, dim]
        debug: dict with keys debug0/debug1/debug2/debug3, each shaped
               [num_m_blocks, num_n_blocks, block_m, *]
    """
    assert q.ndim == 2 and k.ndim == 2 and v.ndim == 2, "q/k/v must be 2D tensors"
    assert q.shape == k.shape == v.shape, "q/k/v must have the same shape"

    seq_len_local, dim_local = q.shape
    num_m_blocks = _ceildiv(seq_len_local, block_m)
    num_n_blocks = _ceildiv(seq_len_local, block_n)
    scale = (1.0 / dim_local) ** 0.5

    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)
    v_fp32 = v.to(torch.float32)

    output_fp32 = torch.zeros(
        (seq_len_local, dim_local), device=q.device, dtype=torch.float32
    )

    debug0 = torch.zeros(
        (num_m_blocks, num_n_blocks, block_m, block_n),
        device=q.device,
        dtype=torch.float32,
    )
    debug1 = torch.zeros(
        (num_m_blocks, num_n_blocks, block_m, block_n),
        device=q.device,
        dtype=out_dtype,
    )
    debug2 = torch.zeros(
        (num_m_blocks, num_n_blocks, block_m, dim_local),
        device=q.device,
        dtype=torch.float32,
    )
    debug3 = torch.zeros(
        (num_m_blocks, num_n_blocks, block_m, dim_local),
        device=q.device,
        dtype=torch.float32,
    )

    for m_blk in range(num_m_blocks):
        m_start = m_blk * block_m
        m_end = min(m_start + block_m, seq_len_local)
        real_m = m_end - m_start

        q_block = q_fp32[m_start:m_end, :]
        running_m = torch.full(
            (real_m, 1), float("-inf"), device=q.device, dtype=torch.float32
        )
        running_l = torch.zeros((real_m, 1), device=q.device, dtype=torch.float32)
        running_o = torch.zeros(
            (real_m, dim_local), device=q.device, dtype=torch.float32
        )

        for n_blk in range(num_n_blocks):
            n_start = n_blk * block_n
            n_end = min(n_start + block_n, seq_len_local)
            real_n = n_end - n_start

            k_block = k_fp32[n_start:n_end, :]
            v_block = v_fp32[n_start:n_end, :]

            scores = q_block @ k_block.T
            debug0[m_blk, n_blk, :real_m, :real_n] = scores

            scaled_scores = scores * scale
            local_max = scaled_scores.max(dim=1, keepdim=True).values
            new_max = torch.maximum(running_m, local_max)
            correction = torch.exp(running_m - new_max)

            prob = torch.exp(scaled_scores - new_max)
            prob_sum = prob.sum(dim=1, keepdim=True)

            running_l = running_l * correction + prob_sum
            prob_cast = prob.to(out_dtype)
            debug1[m_blk, n_blk, :real_m, :real_n] = prob_cast

            contrib = prob_cast.to(torch.float32) @ v_block
            debug2[m_blk, n_blk, :real_m, :] = contrib

            running_o = running_o * correction + contrib
            debug3[m_blk, n_blk, :real_m, :] = running_o

            running_m = new_max

        output_fp32[m_start:m_end, :] = running_o / running_l

    debug = {
        "debug0": debug0,
        "debug1": debug1,
        "debug2": debug2,
        "debug3": debug3,
    }
    return output_fp32.to(out_dtype), debug


def _kernel_debug_to_block_view(
    debug_tensor: torch.Tensor, num_m_blocks: int, num_n_blocks: int
):
    return debug_tensor.reshape(
        num_m_blocks, num_n_blocks, debug_tensor.shape[-2], debug_tensor.shape[-1]
    )


@tilelang.jit(target="npuir")
def online_flash_attention(block_M, block_N, dtype="float16", accum_dtype="float32"):
    shape_q = [seq_len, dim]
    shape_k = [seq_len, dim]
    shape_v = [seq_len, dim]
    shape_o = [seq_len, dim]
    block_m = block_M
    block_n = block_N
    BLOCK_SIZE = T.ceildiv(seq_len, block_m)  # 512 / 16 = 32
    multi_buffer = 2
    shape_debug = [BLOCK_SIZE * 32 * block_m, block_n]

    @T.prim_func
    def flash_attention(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        Output: T.Tensor(shape_o, dtype),
        DEBUG0: T.Tensor(shape_debug, accum_dtype),
        DEBUG1: T.Tensor(shape_debug, dtype),
        DEBUG2: T.Tensor(shape_debug, accum_dtype),
        DEBUG3: T.Tensor(shape_debug, accum_dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, vid):
            offset = cid * block_m

            Q_shared = T.alloc_shared([block_m, dim], dtype)
            K_shared = T.alloc_shared([block_n, dim], dtype)
            V_shared = T.alloc_shared([block_n, dim], dtype)

            T.copy(Q[offset, 0], Q_shared, size=[block_m, dim])
            scores = T.alloc_fragment([block_m, block_n], accum_dtype)
            socres_l1 = T.alloc_shared([block_m, block_n], dtype)
            acc_o_l0c = T.alloc_fragment([block_m, dim], accum_dtype)

            scores_ub = T.alloc_shared([block_m // 2, block_n], accum_dtype)
            scores_cast = T.alloc_shared([block_m // 2, block_n], dtype)
            acc_m = T.alloc_shared([block_m // 2, 1], accum_dtype)
            acc_l = T.alloc_shared([block_m // 2, 1], accum_dtype)
            acc_o_ub = T.alloc_shared([block_m // 2, dim], accum_dtype)

            local_max = T.alloc_shared([block_m // 2, 1], accum_dtype)
            local_sum = T.alloc_shared([block_m // 2, 1], accum_dtype)
            new_max = T.alloc_shared([block_m // 2, 1], accum_dtype)
            correction = T.alloc_shared([block_m // 2, 1], accum_dtype)
            tmp = T.alloc_shared([block_m // 2, block_n], accum_dtype)
            tmp1 = T.alloc_shared([block_m // 2, 1], accum_dtype)

            acc_o = T.alloc_shared([block_m // 2, dim], accum_dtype)
            scales = T.alloc_shared([block_m // 2, block_n], accum_dtype)

            Workspace1 = T.alloc_workspace(
                (BLOCK_SIZE, block_m, block_n), accum_dtype, multi_buffer=multi_buffer
            )
            Workspace2 = T.alloc_workspace(
                (BLOCK_SIZE, block_m, block_n), dtype, multi_buffer=multi_buffer
            )
            Workspace3 = T.alloc_workspace(
                (BLOCK_SIZE, block_m, dim), accum_dtype, multi_buffer=multi_buffer
            )

            value_zero = 0
            scale = (1.0 / dim) ** 0.5
            value_min = -T.infinity(accum_dtype)

            T.vbrc(value_zero, acc_l)
            T.vbrc(value_min, acc_m)
            T.vbrc(scale, scales)

            # seq_len = 512 block_n=16  512/16=32
            for k in T.Pipelined(T.ceildiv(seq_len, block_n), num_stages=multi_buffer):
                # cube
                T.copy(K[k * block_n, 0], K_shared, size=[block_n, dim])
                T.gemm(Q_shared, K_shared, scores, initC=True, b_transpose=True)
                # debug
                T.copy(scores, DEBUG0[(cid * 32 + k) * 16, 0], size=[block_m, block_n])
                T.copy(scores, Workspace1[cid, 0, 0], size=[block_m, block_n])

                # vec
                T.copy(
                    Workspace1[cid, vid * block_m // 2, 0],
                    scores_ub,
                    size=[block_m // 2, block_n],
                )
                T.vmul(scores_ub, scales, scores_ub)
                T.reduce_max(scores_ub, local_max, dim=1)
                T.vmax(acc_m, local_max, new_max)
                T.vsub(acc_m, new_max, tmp1)
                T.vexp(tmp1, correction)
                T.vsub(scores_ub, new_max, tmp)
                T.vexp(tmp, scores_ub)
                T.reduce_sum(scores_ub, local_sum, dim=1)
                T.vmul(acc_l, correction, acc_l)
                T.vadd(acc_l, local_sum, acc_l)
                T.vmul(acc_o, correction, acc_o)
                T.vbrc(value_zero, tmp1)
                T.vadd(tmp1, new_max, acc_m)
                T.vcast(scores_ub, scores_cast, round_mode="rint")
                T.copy(
                    scores_cast,
                    Workspace2[cid, vid * block_m // 2, 0],
                    size=[block_m // 2, block_n],
                )
                # debug
                T.copy(
                    scores_cast,
                    DEBUG1[(cid * 32 + k) * 16 + vid * block_m // 2, 0],
                    size=[block_m // 2, block_n],
                )
                T.copy(Workspace2[cid, 0, 0], socres_l1)

                # cube
                T.copy(V[k * block_n, 0], V_shared, size=[block_n, dim])
                T.gemm(socres_l1, V_shared, acc_o_l0c, initC=True)
                # debug
                T.copy(
                    acc_o_l0c, DEBUG2[(cid * 32 + k) * 16, 0], size=[block_m, block_n]
                )
                T.copy(acc_o_l0c, Workspace3[cid, 0, 0], size=[block_m, dim])

                # vec
                T.copy(
                    Workspace3[cid, vid * block_m // 2, 0],
                    acc_o_ub,
                    size=[block_m // 2, dim],
                )
                T.vadd(acc_o, acc_o_ub, acc_o)
                # debug
                T.copy(
                    acc_o,
                    DEBUG3[(cid * 32 + k) * 16 + vid * block_m // 2, 0],
                    size=[block_m // 2, block_n],
                )

            T.vdiv(acc_o, acc_l, acc_o)
            O_cast = T.alloc_shared([block_m // 2, dim], dtype)
            T.vcast(acc_o, O_cast, round_mode="rint")
            real_m = T.min(block_m // 2, seq_len - cid * block_m - vid * block_m // 2)
            T.copy(
                O_cast,
                Output[cid * block_m + vid * block_m // 2, 0],
                size=[real_m, dim],
            )

    return flash_attention


def main():
    # In the futrue, Developer mode and Expert Mode will transition smoothly without
    # requiring explicit declarations.
    # os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    block_m = 16
    block_n = 16
    kernel = online_flash_attention(block_m, block_n)

    q = torch.randn((seq_len, dim), dtype=torch.float16).npu()
    k = torch.randn((seq_len, dim), dtype=torch.float16).npu()
    v = torch.randn((seq_len, dim), dtype=torch.float16).npu()
    num_m_blocks = _ceildiv(seq_len, block_m)
    num_n_blocks = _ceildiv(seq_len, block_n)
    debug0 = torch.zeros(
        (num_m_blocks * num_n_blocks, block_m, block_n), dtype=torch.float32
    ).npu()
    debug1 = torch.zeros(
        (num_m_blocks * num_n_blocks, block_m, block_n), dtype=torch.float16
    ).npu()
    debug2 = torch.zeros(
        (num_m_blocks * num_n_blocks, block_m, block_n), dtype=torch.float32
    ).npu()
    debug3 = torch.zeros(
        (num_m_blocks * num_n_blocks, block_m, block_n), dtype=torch.float32
    ).npu()
    output = torch.zeros((seq_len, dim), dtype=torch.float16).npu()

    scale = (1.0 / dim) ** 0.5
    ref_output = (
        torch.nn.functional.softmax((q @ k.T).to(torch.float32) * scale, dim=-1).to(
            torch.float16
        )
        @ v
    )
    blocked_ref_output, blocked_debug = torch_online_flash_attention_blocked_debug(
        q, k, v, block_m=block_m, block_n=block_n, out_dtype=torch.float16
    )
    torch.testing.assert_close(blocked_ref_output, ref_output, rtol=1e-2, atol=1e-2)
    # output = kernel(q, k, v)
    kernel(q, k, v, output, debug0, debug1, debug2, debug3)

    kernel_debug0 = _kernel_debug_to_block_view(debug0, num_m_blocks, num_n_blocks)
    kernel_debug1 = _kernel_debug_to_block_view(debug1, num_m_blocks, num_n_blocks)
    kernel_debug2 = _kernel_debug_to_block_view(debug2, num_m_blocks, num_n_blocks)
    kernel_debug3 = _kernel_debug_to_block_view(debug3, num_m_blocks, num_n_blocks)

    print("output:")
    print(output)
    print("ref_output:")
    print(ref_output)

    # Check final output correctness (wrapped so debug block still runs on failure)
    output_ok = True
    for label, a, b in [
        ("ref vs kernel", ref_output, output),
        ("blocked_ref vs kernel", blocked_ref_output, output),
    ]:
        try:
            torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-2)
        except AssertionError as e:
            print(f"[FAIL] final output mismatch ({label}): {e}")
            output_ok = False

    # Per-block debug comparison —— locate which cid / iteration diverges
    DIFF_THRESHOLD = 1e-2
    torch.set_printoptions(precision=5, sci_mode=True, linewidth=160)

    d0_all = (blocked_debug["debug0"] - kernel_debug0).abs()
    d1_all = (
        blocked_debug["debug1"].to(torch.float32) - kernel_debug1.to(torch.float32)
    ).abs()
    d2_all = (blocked_debug["debug2"][..., :block_n] - kernel_debug2).abs()
    d3_all = (blocked_debug["debug3"][..., :block_n] - kernel_debug3).abs()

    print("\n===== DEBUG CHECK SUMMARY (per block) =====")
    print(
        f"{'cid':>4} {'iter':>4} | "
        f"{'D0(scores)':>12} {'D1(prob)':>12} "
        f"{'D2(contrib)':>12} {'D3(acc_o)':>12}"
    )
    print("-" * 68)

    any_fail = False
    for m_blk in range(num_m_blocks):
        for n_blk in range(num_n_blocks):
            max_d0 = d0_all[m_blk, n_blk].max().item()
            max_d1 = d1_all[m_blk, n_blk].max().item()
            max_d2 = d2_all[m_blk, n_blk].max().item()
            max_d3 = d3_all[m_blk, n_blk].max().item()
            fail = max(max_d0, max_d1, max_d2, max_d3) > DIFF_THRESHOLD
            any_fail = any_fail or fail
            tag = " <-- FAIL" if fail else ""
            print(
                f"{m_blk:>4} {n_blk:>4} | "
                f"{max_d0:>12.4e} {max_d1:>12.4e} "
                f"{max_d2:>12.4e} {max_d3:>12.4e}{tag}"
            )
    print("-" * 68)

    def print_thresholded_error_matrices(name: str, diff_all: torch.Tensor):
        print(f"\n===== {name} thresholded error matrices (cid, iter) =====")
        printed = False
        for m_blk in range(num_m_blocks):
            for n_blk in range(num_n_blocks):
                diff_block = diff_all[m_blk, n_blk]
                masked = torch.where(
                    diff_block > DIFF_THRESHOLD,
                    diff_block,
                    torch.zeros_like(diff_block),
                )
                max_val = masked.max().item()
                if max_val > 0:
                    printed = True
                    print(f"[{name}] cid={m_blk}, iter={n_blk}, max_diff={max_val:.6e}")
                    print(masked)
        if not printed:
            print(f"[{name}] all blocks within threshold {DIFF_THRESHOLD}")

    print_thresholded_error_matrices("D0(scores)", d0_all)
    print_thresholded_error_matrices("D1(prob)", d1_all)
    print_thresholded_error_matrices("D2(contrib)", d2_all)
    print_thresholded_error_matrices("D3(acc_o)", d3_all)

    if any_fail or not output_ok:
        print("[RESULT] Precision divergence detected. See FAIL rows above.")
    else:
        print("[RESULT] All blocks passed.")


if __name__ == "__main__":
    torch.manual_seed(88888888)
    main()
