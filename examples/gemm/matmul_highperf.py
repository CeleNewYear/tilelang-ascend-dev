# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import torch
import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

# shape of L1 is 512KB
# multibuffer can hide the latency of L1 load, a single buffer use 256KB
# shape of A1_L1 and B1_l1 is 128KB = 65536 fp16 elements = 256  * 256 (Limited by L0C size use 128 * 256)
# shape of A2_L1 and B2_l1 is 128KB = 65536 fp16 elements = 256  * 256 (Limited by L0C size use 128 * 256)
# shape of C1_L0C is 64KB = 16384 fp32 elements = 128 * 128

M = 1024
N = 1024
K = 256


@tilelang.jit(target="npuir")
def matmul(block_M, block_N, K_L1, dtype="float16", accum_dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                for by_idx in T.serial(n_num):
                    bx = cid * block_M
                    by = by_idx * block_N
                    A1_L1 = T.alloc_L1([block_M, K], dtype)
                    B1_L1 = T.alloc_L1([K, block_N], dtype)
                    C1_L0C = T.alloc_L0C([block_M, block_N], accum_dtype)

                    T.load_nd2nz(A[bx, 0], A1_L1, [block_M, K])
                    T.load_nd2nz(B[0, by], B1_L1, [K, block_N])
                    T.gemm(
                        A1_L1,
                        B1_L1,
                        C1_L0C,
                        initC=True,
                        b_transpose=False,
                        size=[block_M, K, block_N],
                    )
                    T.store_fixpipe(
                        C1_L0C, C[bx, by], size=[block_M, block_N], enable_nz2nd=True
                    )

    return main


def test_mat_mul():
    func = matmul(128, 128, 256)
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()
    c = torch.randn(M, N).float().npu()

    func(a, b, c)
    print(c)

    ref_c = a.to(torch.float32) @ b.to(torch.float32)
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    test_mat_mul()
