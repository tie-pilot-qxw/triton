import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from typing import Optional
from triton.tools.tensor_descriptor import TensorDescriptor
import triton.profiler.language as pl
import triton.profiler as proton
from triton.testing import do_bench

DEVICE = triton.runtime.driver.active.get_active_torch_device()

M, N, K = (8192, 8192, 8192)

flops = 2.0 * M * N * K  # FLOPs for matrix multiplication

torch.manual_seed(0)

a = torch.randn((M, K), dtype=torch.float16, device=DEVICE)
b = torch.randn((K, N), dtype=torch.float16, device=DEVICE)

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_cdna2():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def alloc_fn(size: int, align: int, stream: Optional[int]):
    assert align == 128
    assert stream == 0
    return torch.empty(size, dtype=torch.int8, device=DEVICE)

triton.set_allocator(alloc_fn)

def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BM"]
    BLOCK_N = nargs["BN"]
    BLOCK_K = nargs["BK"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["a_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N]

@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

rtol = 1e-2 if is_hip_cdna2() else 0

#  =============== PyTorch version of the GEMM+GELU kernel ===============
def pt_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))

ms = do_bench(lambda: pt_gelu(torch.matmul(a, b)))
# Calculate throughput metrics
tflops = flops / (ms * 1e-3) / 1e12  # TFLOPS
print(f"PyTorch Time: {ms:.3f} ms")
print(f"PyTorch TFLOPS: {tflops:.2f}")

output_ref = pt_gelu(torch.matmul(a, b))

#  =============== TLX WS version of the GEMM+GELU kernel ===============
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BM": 128,
                "BN": 128,
                "BK": 64,
                "GROUP_SIZE_M": 8,
                "NUM_STAGES": 4,
                "NUM_MMA_WARPS": 8,
                "NUM_MMA_GROUPS": 2,
                "EPILOGUE_SUBTILE": False,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=matmul_tma_set_block_size_hook
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@triton.jit
def tlx_ws_kernel(
    a_desc, b_desc, c_desc,  #
    M, N, K,  #
    BM: tl.constexpr,  #
    BN: tl.constexpr,  #
    BK: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,  #
    NUM_MMA_WARPS: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
):
    # Descriptor
    BLOCK_M_SPLIT: tl.constexpr = BM // NUM_MMA_GROUPS

    a = tlx.local_alloc((BLOCK_M_SPLIT, BK), tlx.dtype_of(a_desc), NUM_STAGES * NUM_MMA_GROUPS)
    b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)

    bars_empty_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1)
    bars_full_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1)
    bars_empty_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=NUM_MMA_GROUPS)
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    # Warp specilization
    with tlx.async_tasks():
        # Producer (async load)
        with tlx.async_task("default"):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            offset_am = pid_m * BM
            offset_bn = pid_n * BN

            p = 1
            for k in range(0, tl.cdiv(K, BK)):
                buf = k % NUM_STAGES
                offset_k = k * BK

                # Async load to a[buf]
                empty_a_1st = tlx.local_view(bars_empty_a, buf)  # mbar
                full_a_1st = tlx.local_view(bars_full_a, buf)  # mbar
                tlx.barrier_wait(bar=empty_a_1st, phase=p)  # EmptyBar A1 wait
                tlx.barrier_expect_bytes(full_a_1st, BLOCK_M_SPLIT * BK * 2)
                data_a_1st = tlx.local_view(a, buf)  # smem data
                tlx.async_descriptor_load(
                    a_desc,
                    data_a_1st,
                    [offset_am, offset_k],
                    full_a_1st)

                # Async load to b[buf]
                empty_b = tlx.local_view(bars_empty_b, buf)
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_wait(bar=empty_b, phase=p)
                tlx.barrier_expect_bytes(full_b, BN * BK * 2)
                data_b = tlx.local_view(b, buf)
                tlx.async_descriptor_load(
                    b_desc,
                    data_b,
                    [offset_k, offset_bn],
                    full_b)

                # Async load to a[buf+NUM_STAGES]
                empty_a_2nd = tlx.local_view(bars_empty_a, buf+NUM_STAGES)
                full_a_2nd = tlx.local_view(bars_full_a, buf+NUM_STAGES)
                tlx.barrier_wait(bar=empty_a_2nd, phase=p)
                tlx.barrier_expect_bytes(bar=full_a_2nd, size=BLOCK_M_SPLIT * BK * 2)
                data_a_2nd = tlx.local_view(a, buf+NUM_STAGES)  # smem data
                tlx.async_descriptor_load(
                    a_desc,
                    data_a_2nd,
                    [offset_am + BLOCK_M_SPLIT, offset_k],
                    full_a_2nd)

                # Flip phase after every NUM_STAGES iterations finish
                p = p ^ (buf == (NUM_STAGES-1))

        # consumers (wgmma + async store)
        with tlx.async_task(num_warps=4, replicate=2):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            offset_am = pid_m * BM
            offset_bn = pid_n * BN

            p = 0
            acc = tl.zeros([BM//2, BN], dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BK)):
                buf = k % NUM_STAGES

                # Wait for TMA load
                full_a = tlx.local_view(bars_full_a, buf + NUM_STAGES * tlx.async_task_replica_id()) # noqa
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_wait(bar=full_a, phase=p)
                tlx.barrier_wait(bar=full_b, phase=p)

                # async_dot
                data_a = tlx.local_view(a, buf + NUM_STAGES * tlx.async_task_replica_id()) # noqa
                data_b = tlx.local_view(b, buf)
                acc = tlx.async_dot(
                    data_a,
                    data_b,
                    acc,
                )
                # async_wait
                acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                # Release buffers
                empty_a = tlx.local_view(bars_empty_a, buf + NUM_STAGES * tlx.async_task_replica_id()) # noqa
                empty_b = tlx.local_view(bars_empty_b, buf)
                tlx.barrier_arrive(empty_a)  # EmptyBar A1 arrive
                tlx.barrier_arrive(empty_b)

                # Flip phase after every NUM_STAGES iterations finish
                p = p ^ (buf == (NUM_STAGES-1))
 
            offset_cm = offset_am + BLOCK_M_SPLIT * tlx.async_task_replica_id()
            act = gelu(acc)
            res = act.to(tlx.dtype_of(c_desc))
            c_desc.store([offset_cm, offset_bn], res)  # noqa


def tlx_ws(a, b,):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Illegal dimensions of input operands"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    (M, N, K) = (a.shape[0], b.shape[1], a.shape[1])
    c = torch.zeros((M, N), dtype=torch.float16, device=DEVICE, )

    dummy_block = [1, 1]
    desc_in_1 = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    desc_in_2 = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    desc_out = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    grid = lambda META: (  # noqa E731
        triton.cdiv(M, META['BM']) * triton.cdiv(N, META['BN']),
    )
    tlx_ws_kernel[grid](
        desc_in_1, desc_in_2, desc_out,  #
        M, N, K,  #
    )
    return c

output_tlx_ws = tlx_ws(a, b)
if torch.allclose(output_tlx_ws, output_ref, atol=1e-2, rtol=rtol):
    print("✅ TLX-WS and Torch match")
else:
    print("❌ TLX-WS and Torch differ")

ms = do_bench(lambda: tlx_ws(a, b))
# Calculate throughput metrics
tflops = flops / (ms * 1e-3) / 1e12  # TFLOPS
print(f"TLX-WS Time: {ms:.3f} ms")
print(f"TLX-WS TFLOPS: {tflops:.2f}")

#  =============== TLX Persistent WS version of the GEMM+GELU kernel ===============
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BM": 128,
                "BN": 128,
                "BK": 64,
                "GROUP_SIZE_M": 8,
                "NUM_STAGES": 4,
                "NUM_MMA_WARPS": 8,
                "NUM_MMA_GROUPS": 2,
                "EPILOGUE_SUBTILE": False,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=matmul_tma_set_block_size_hook
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@triton.jit
def tlx_ws_persist_kernel(
    a_desc, b_desc, c_desc,  #
    M, N, K,  #
    BM: tl.constexpr,  #
    BN: tl.constexpr,  #
    BK: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,  #
    NUM_MMA_WARPS: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
):
    # Descriptor
    BLOCK_M_SPLIT: tl.constexpr = BM // NUM_MMA_GROUPS

    a = tlx.local_alloc((BLOCK_M_SPLIT, BK), tlx.dtype_of(a_desc), NUM_STAGES * NUM_MMA_GROUPS)
    b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)

    bars_empty_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1)
    bars_full_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1)
    bars_empty_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=NUM_MMA_GROUPS)
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    # Warp specilization
    with tlx.async_tasks():
        # Producer (async load)
        with tlx.async_task("default"):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            k_tiles = tl.cdiv(K, BK)
            num_tiles = num_pid_m * num_pid_n

            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            processed_k_iters = 0
            p = 1
            for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)

                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                for k in range(0, k_tiles):
                    buf = (processed_k_iters + k) % NUM_STAGES
                    offset_k = k * BK

                    # Async load to a[buf]
                    empty_a_1st = tlx.local_view(bars_empty_a, buf)  # mbar
                    full_a_1st = tlx.local_view(bars_full_a, buf)  # mbar
                    tlx.barrier_wait(bar=empty_a_1st, phase=p)  # EmptyBar A1 wait
                    tlx.barrier_expect_bytes(full_a_1st, BLOCK_M_SPLIT * BK * 2)
                    data_a_1st = tlx.local_view(a, buf)  # smem data
                    tlx.async_descriptor_load(
                        a_desc,
                        data_a_1st,
                        [offset_am, offset_k],
                        full_a_1st)

                    # Async load to b[buf]
                    empty_b = tlx.local_view(bars_empty_b, buf)
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(bar=empty_b, phase=p)
                    tlx.barrier_expect_bytes(full_b, BN * BK * 2)
                    data_b = tlx.local_view(b, buf)
                    tlx.async_descriptor_load(
                        b_desc,
                        data_b,
                        [offset_k, offset_bn],
                        full_b)

                    # Async load to a[buf+NUM_STAGES]
                    empty_a_2nd = tlx.local_view(bars_empty_a, buf+NUM_STAGES)
                    full_a_2nd = tlx.local_view(bars_full_a, buf+NUM_STAGES)
                    tlx.barrier_wait(bar=empty_a_2nd, phase=p)
                    tlx.barrier_expect_bytes(bar=full_a_2nd, size=BLOCK_M_SPLIT * BK * 2)
                    data_a_2nd = tlx.local_view(a, buf+NUM_STAGES)  # smem data
                    tlx.async_descriptor_load(
                        a_desc,
                        data_a_2nd,
                        [offset_am + BLOCK_M_SPLIT, offset_k],
                        full_a_2nd)

                    # Flip phase after every NUM_STAGES iterations finish
                    p = p ^ (buf == (NUM_STAGES-1))
                
                # wait for last mma to complete
                last_buf = (processed_k_iters + k_tiles - 1) % NUM_STAGES
                last_empty_a1 = tlx.local_view(bars_empty_a, last_buf) # noqa
                last_empty_a2 = tlx.local_view(bars_empty_a, last_buf + NUM_STAGES) # noqa
                last_empty_b = tlx.local_view(bars_empty_b, last_buf)
                last_dot_phase = p ^ (last_buf == NUM_STAGES - 1)
                tlx.barrier_wait(last_empty_a1, last_dot_phase)
                tlx.barrier_wait(last_empty_a2, last_dot_phase)
                tlx.barrier_wait(last_empty_b, last_dot_phase)
                processed_k_iters += k_tiles

        # consumers (wgmma + async store)
        with tlx.async_task(num_warps=4, replicate=2):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            k_tiles = tl.cdiv(K, BK)
            num_tiles = num_pid_m * num_pid_n

            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            processed_k_iters = 0
            p = 0
            for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
                acc = tl.zeros([BM//2, BN], dtype=tl.float32)
                for k in range(0, k_tiles):
                    buf = (processed_k_iters + k) % NUM_STAGES

                    # Wait for TMA load
                    full_a = tlx.local_view(bars_full_a, buf + NUM_STAGES * tlx.async_task_replica_id()) # noqa
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(bar=full_a, phase=p)
                    tlx.barrier_wait(bar=full_b, phase=p)

                    # async_dot
                    data_a = tlx.local_view(a, buf + NUM_STAGES * tlx.async_task_replica_id()) # noqa
                    data_b = tlx.local_view(b, buf)
                    acc = tlx.async_dot(
                        data_a,
                        data_b,
                        acc,
                    )
                    # async_wait
                    acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                    # Release buffers
                    empty_a = tlx.local_view(bars_empty_a, buf + NUM_STAGES * tlx.async_task_replica_id()) # noqa
                    empty_b = tlx.local_view(bars_empty_b, buf)
                    tlx.barrier_arrive(empty_a)  # EmptyBar A1 arrive
                    tlx.barrier_arrive(empty_b)

                    # Flip phase after every NUM_STAGES iterations finish
                    p = p ^ (buf == (NUM_STAGES-1))
                act = gelu(acc)
                res = act.to(tlx.dtype_of(c_desc))
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_bn_c = pid_n * BN
                offs_am_c = pid_m * BM
                offset_cm = offs_am_c + BLOCK_M_SPLIT * tlx.async_task_replica_id()

                c_desc.store([offset_cm, offs_bn_c], res)  # noqa
                processed_k_iters += k_tiles


def tlx_ws_persist(a, b,):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Illegal dimensions of input operands"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    (M, N, K) = (a.shape[0], b.shape[1], a.shape[1])
    c = torch.zeros((M, N), dtype=torch.float16, device=DEVICE, )

    dummy_block = [1, 1]
    desc_in_1 = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    desc_in_2 = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    desc_out = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"])), )

    tlx_ws_persist_kernel[grid](
        desc_in_1, desc_in_2, desc_out,  #
        M, N, K,  #
        NUM_SMS=NUM_SMS,
    )
    return c


output_tlx_ws_persist = tlx_ws_persist(a, b)
if torch.allclose(output_tlx_ws_persist, output_ref, atol=1e-2, rtol=rtol):
    print("✅ TLX-WS-persist and Torch match")
else:
    print("❌ TLX-WS-persist and Torch differ")

ms = do_bench(lambda: tlx_ws_persist(a, b))
# Calculate throughput metrics
tflops = flops / (ms * 1e-3) / 1e12  # TFLOPS
print(f"TLX-WS-persist Time: {ms:.3f} ms")
print(f"TLX-WS-persist TFLOPS: {tflops:.2f}")

#  =============== TLX Persistent WS PingPongV1 version of the GEMM+GELU kernel ===============
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BM": 128,
                "BN": 128,
                "BK": 64,
                "GROUP_SIZE_M": 8,
                "NUM_STAGES": 4, # must be 2, 4, 8
                "NUM_MMA_WARPS": 8, # fixed
                "NUM_MMA_GROUPS": 2, # fixed
                "EPILOGUE_SUBTILE": False,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=matmul_tma_set_block_size_hook
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@triton.jit
def tlx_ws_pingpong_v1_kernel(
    a_desc, b_desc, c_desc,  #
    M, N, K,  #
    BM: tl.constexpr,  #
    BN: tl.constexpr,  #
    BK: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,  #
    NUM_MMA_WARPS: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
):
    # Descriptor
    BLOCK_M_SPLIT: tl.constexpr = BM // NUM_MMA_GROUPS

    a = tlx.local_alloc((BLOCK_M_SPLIT, BK), tlx.dtype_of(a_desc), NUM_STAGES * NUM_MMA_GROUPS)
    b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)

    bars_empty_a1 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_full_a1 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_empty_a2 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_full_a2 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_empty_b1 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_full_b1 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_empty_b2 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_full_b2 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    gemm_bars_a1 = tlx.alloc_barriers(num_barriers=1, arrive_count=2)
    gemm_bars_a2 = tlx.alloc_barriers(num_barriers=1, arrive_count=2)

    # Warp specilization
    with tlx.async_tasks():
        # Producer (async load)
        with tlx.async_task("default"):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            k_tiles = tl.cdiv(K, BK)
            num_tiles = num_pid_m * num_pid_n

            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)

                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                p = 1
                for k in range(0, k_tiles):
                    buf = k % NUM_STAGES
                    offset_k = k * BK

                    # Async load to a[buf]
                    empty_a_1st = tlx.local_view(bars_empty_a1, buf)  # mbar
                    full_a_1st = tlx.local_view(bars_full_a1, buf)  # mbar
                    tlx.barrier_wait(bar=empty_a_1st, phase=p)  # EmptyBar A1 wait
                    tlx.barrier_expect_bytes(full_a_1st, BLOCK_M_SPLIT * BK * 2)
                    data_a_1st = tlx.local_view(a, buf)  # smem data
                    tlx.async_descriptor_load(
                        a_desc,
                        data_a_1st,
                        [offset_am, offset_k],
                        full_a_1st)

                    # Async load to b[buf]
                    empty_b = tlx.local_view(bars_empty_b1, buf)
                    full_b = tlx.local_view(bars_full_b1, buf)
                    tlx.barrier_wait(bar=empty_b, phase=p)
                    tlx.barrier_expect_bytes(full_b, BN * BK * 2)
                    data_b = tlx.local_view(b, buf)
                    tlx.async_descriptor_load(
                        b_desc,
                        data_b,
                        [offset_k, offset_bn],
                        full_b)
                    # Flip phase after every NUM_STAGES iterations finish
                    p = p ^ (buf == (NUM_STAGES-1))
                empty_a_1st = tlx.local_view(bars_empty_a1, 1)  # mbar
                empty_b_1 = tlx.local_view(bars_empty_b1, 1)
                last_dot_phase = 1
                tlx.barrier_wait(bar=empty_a_1st, phase=last_dot_phase)  # EmptyBar A1 wait
                tlx.barrier_wait(bar=empty_b_1, phase=last_dot_phase)

                p = 1
                for k in range(0, k_tiles):
                    buf = k % NUM_STAGES
                    offset_k = k * BK

                    # Async load to a[buf+NUM_STAGES]
                    empty_a_2nd = tlx.local_view(bars_empty_a2, buf)
                    full_a_2nd = tlx.local_view(bars_full_a2, buf)
                    tlx.barrier_wait(bar=empty_a_2nd, phase=p)
                    tlx.barrier_expect_bytes(bar=full_a_2nd, size=BLOCK_M_SPLIT * BK * 2)
                    data_a_2nd = tlx.local_view(a, buf + NUM_STAGES)  # smem data
                    tlx.async_descriptor_load(
                        a_desc,
                        data_a_2nd,
                        [offset_am + BLOCK_M_SPLIT, offset_k],
                        full_a_2nd)

                    # Async load to b[buf]
                    empty_b = tlx.local_view(bars_empty_b2, buf)
                    full_b = tlx.local_view(bars_full_b2, buf)
                    tlx.barrier_wait(bar=empty_b, phase=p)
                    tlx.barrier_expect_bytes(full_b, BN * BK * 2)
                    data_b = tlx.local_view(b, buf)
                    tlx.async_descriptor_load(
                        b_desc,
                        data_b,
                        [offset_k, offset_bn],
                        full_b)

                    # Flip phase after every NUM_STAGES iterations finish
                    p = p ^ (buf == (NUM_STAGES-1))
                

                empty_a_2nd = tlx.local_view(bars_empty_a2, 1)
                empty_b_2 = tlx.local_view(bars_empty_b2, 1)
                last_dot_phase = 1
                tlx.barrier_wait(bar=empty_a_2nd, phase=last_dot_phase)
                tlx.barrier_wait(bar=empty_b_2, phase=last_dot_phase)

        # consumers (wgmma + async store)
        # first half of the tile A
        with tlx.async_task(num_warps=4):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            k_tiles = tl.cdiv(K, BK)
            num_tiles = num_pid_m * num_pid_n

            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            p_gemm = 0
            for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
                p = 0
                acc = tl.zeros([BM//2, BN], dtype=tl.float32)
                for k in range(0, k_tiles):
                    buf = k % NUM_STAGES

                    # Wait for TMA load
                    full_a = tlx.local_view(bars_full_a1, buf) # noqa
                    full_b = tlx.local_view(bars_full_b1, buf)
                    tlx.barrier_wait(bar=full_a, phase=p)
                    tlx.barrier_wait(bar=full_b, phase=p)

                    # async_dot
                    data_a = tlx.local_view(a, buf) # noqa
                    data_b = tlx.local_view(b, buf)
                    acc = tlx.async_dot(
                        data_a,
                        data_b,
                        acc,
                    )
                    # async_wait
                    acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                    # Release buffers
                    empty_a = tlx.local_view(bars_empty_a1, buf) # noqa
                    empty_b = tlx.local_view(bars_empty_b1, buf)
                    tlx.barrier_arrive(empty_a)  # EmptyBar A1 arrive
                    tlx.barrier_arrive(empty_b)

                    # Flip phase after every NUM_STAGES iterations finish
                    p = p ^ (buf == (NUM_STAGES-1))

                gemm_bar1 = tlx.local_view(gemm_bars_a1, 0)
                tlx.barrier_arrive(gemm_bar1)
                tlx.barrier_wait(bar=gemm_bar1, phase=p_gemm)

                act = gelu(acc)

                gemm_bar2 = tlx.local_view(gemm_bars_a2, 0)
                tlx.barrier_arrive(gemm_bar2)
                tlx.barrier_wait(bar=gemm_bar2, phase=p_gemm)

                res = act.to(tlx.dtype_of(c_desc))
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_bn_c = pid_n * BN
                offs_am_c = pid_m * BM
                offset_cm = offs_am_c
                c_desc.store([offset_cm, offs_bn_c], res)  # noqa

                p_gemm = p_gemm ^ 1

        # consumers (wgmma + async store)
        # second half of the tile A
        with tlx.async_task(num_warps=4):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            k_tiles = tl.cdiv(K, BK)
            num_tiles = num_pid_m * num_pid_n

            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            p_gemm = 0
            gemm_bar1 = tlx.local_view(gemm_bars_a1, 0)
            tlx.barrier_arrive(gemm_bar1)
            tlx.barrier_wait(bar=gemm_bar1, phase=p_gemm)
            for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
                p = 0
                acc = tl.zeros([BM//2, BN], dtype=tl.float32)
                for k in range(0, k_tiles):
                    buf = k % NUM_STAGES

                    # Wait for TMA load
                    full_a = tlx.local_view(bars_full_a2, buf) # noqa
                    full_b = tlx.local_view(bars_full_b2, buf)
                    tlx.barrier_wait(bar=full_a, phase=p)
                    tlx.barrier_wait(bar=full_b, phase=p)

                    # async_dot
                    data_a = tlx.local_view(a, buf + NUM_STAGES) # noqa
                    data_b = tlx.local_view(b, buf)
                    acc = tlx.async_dot(
                        data_a,
                        data_b,
                        acc,
                    )
                    # async_wait
                    acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                    # Release buffers
                    empty_a = tlx.local_view(bars_empty_a2, buf) # noqa
                    empty_b = tlx.local_view(bars_empty_b2, buf)
                    tlx.barrier_arrive(empty_a)  # EmptyBar A1 arrive
                    tlx.barrier_arrive(empty_b)

                    # Flip phase after every NUM_STAGES iterations finish
                    p = p ^ (buf == (NUM_STAGES-1))

                gemm_bar2 = tlx.local_view(gemm_bars_a2, 0)
                tlx.barrier_arrive(gemm_bar2)
                tlx.barrier_wait(bar=gemm_bar2, phase=p_gemm)
                
                act = gelu(acc)

                if tile_id < num_tiles - NUM_SMS:
                    gemm_bar1 = tlx.local_view(gemm_bars_a1, 0)
                    tlx.barrier_arrive(gemm_bar1)
                    tlx.barrier_wait(bar=gemm_bar1, phase=p_gemm ^ 1)

                res = act.to(tlx.dtype_of(c_desc))
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_bn_c = pid_n * BN
                offs_am_c = pid_m * BM
                offset_cm = offs_am_c + BLOCK_M_SPLIT
                c_desc.store([offset_cm, offs_bn_c], res)  # noqa

                p_gemm = p_gemm ^ 1


def tlx_ws_pingpong_v1(a, b,):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Illegal dimensions of input operands"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    (M, N, K) = (a.shape[0], b.shape[1], a.shape[1])
    c = torch.zeros((M, N), dtype=torch.float16, device=DEVICE, )

    dummy_block = [1, 1]
    desc_in_1 = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    desc_in_2 = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    desc_out = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"])), )

    tlx_ws_pingpong_v1_kernel[grid](
        desc_in_1, desc_in_2, desc_out,  #
        M, N, K,  #
        NUM_SMS=NUM_SMS,
    )
    return c

output_tlx_ws_pingpong_v1 = tlx_ws_pingpong_v1(a, b)
if torch.allclose(output_tlx_ws_pingpong_v1, output_ref, atol=1e-2, rtol=rtol):
    print("✅ TLX-WS-pingpong-V1 and Torch match")
else:
    print("❌ TLX-WS-pingpong-V1 and Torch differ")

ms = do_bench(lambda: tlx_ws_pingpong_v1(a, b))
# Calculate throughput metrics
tflops = flops / (ms * 1e-3) / 1e12  # TFLOPS
print(f"TLX-WS-pingpong-V1 Time: {ms:.3f} ms")
print(f"TLX-WS-pingpong-V1 TFLOPS: {tflops:.2f}")

#  =============== TLX Persistent WS PingPongV2 version of the GEMM+GELU kernel ===============
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BM": 128,
                "BN": 128,
                "BK": 64,
                "GROUP_SIZE_M": 8,
                "NUM_STAGES": 4, # must be 2 or 4 or 8
                "NUM_MMA_WARPS": 8, # fixed
                "NUM_MMA_GROUPS": 2, # fixed
                "EPILOGUE_SUBTILE": False,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=matmul_tma_set_block_size_hook
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@triton.jit
def tlx_ws_pingpong_v2_kernel(
    a_desc, b_desc, c_desc,  #
    M, N, K,  #
    BM: tl.constexpr,  #
    BN: tl.constexpr,  #
    BK: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,  #
    NUM_MMA_WARPS: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
):
    # Descriptor
    BLOCK_M_SPLIT: tl.constexpr = BM // NUM_MMA_GROUPS

    a = tlx.local_alloc((BLOCK_M_SPLIT, BK), tlx.dtype_of(a_desc), NUM_STAGES * NUM_MMA_GROUPS)
    b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES * NUM_MMA_GROUPS)

    bars_empty_a1 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_full_a1 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_empty_a2 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_full_a2 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_empty_b1 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_full_b1 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_empty_b2 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_full_b2 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    # Warp specilization
    with tlx.async_tasks():
        # Producer (async load)
        with tlx.async_task("default"):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            k_tiles = tl.cdiv(K, BK)
            num_tiles = num_pid_m * num_pid_n

            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)

                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                p = 1
                for k in range(0, k_tiles):
                    buf = k % NUM_STAGES
                    offset_k = k * BK

                    # Async load to a[buf]
                    empty_a_1st = tlx.local_view(bars_empty_a1, buf)  # mbar
                    full_a_1st = tlx.local_view(bars_full_a1, buf)  # mbar
                    tlx.barrier_wait(bar=empty_a_1st, phase=p)  # EmptyBar A1 wait
                    tlx.barrier_expect_bytes(full_a_1st, BLOCK_M_SPLIT * BK * 2)
                    data_a_1st = tlx.local_view(a, buf)  # smem data
                    tlx.async_descriptor_load(
                        a_desc,
                        data_a_1st,
                        [offset_am, offset_k],
                        full_a_1st)

                    # Async load to b[buf]
                    empty_b = tlx.local_view(bars_empty_b1, buf)
                    full_b = tlx.local_view(bars_full_b1, buf)
                    tlx.barrier_wait(bar=empty_b, phase=p)
                    tlx.barrier_expect_bytes(full_b, BN * BK * 2)
                    data_b = tlx.local_view(b, buf)
                    tlx.async_descriptor_load(
                        b_desc,
                        data_b,
                        [offset_k, offset_bn],
                        full_b)
                    # Flip phase after every NUM_STAGES iterations finish
                    p = p ^ (buf == (NUM_STAGES-1))
                empty_a_1st = tlx.local_view(bars_empty_a1, 1)  # mbar
                empty_b_1 = tlx.local_view(bars_empty_b1, 1)
                last_dot_phase = 1
                tlx.barrier_wait(bar=empty_a_1st, phase=last_dot_phase)  # EmptyBar A1 wait
                tlx.barrier_wait(bar=empty_b_1, phase=last_dot_phase)

                p = 1
                for k in range(0, k_tiles):
                    buf = k % NUM_STAGES
                    offset_k = k * BK

                    # Async load to a[buf+NUM_STAGES]
                    empty_a_2nd = tlx.local_view(bars_empty_a2, buf)
                    full_a_2nd = tlx.local_view(bars_full_a2, buf)
                    tlx.barrier_wait(bar=empty_a_2nd, phase=p)
                    tlx.barrier_expect_bytes(bar=full_a_2nd, size=BLOCK_M_SPLIT * BK * 2)
                    data_a_2nd = tlx.local_view(a, buf + NUM_STAGES)  # smem data
                    tlx.async_descriptor_load(
                        a_desc,
                        data_a_2nd,
                        [offset_am + BLOCK_M_SPLIT, offset_k],
                        full_a_2nd)

                    # Async load to b[buf]
                    empty_b = tlx.local_view(bars_empty_b2, buf)
                    full_b = tlx.local_view(bars_full_b2, buf)
                    tlx.barrier_wait(bar=empty_b, phase=p)
                    tlx.barrier_expect_bytes(full_b, BN * BK * 2)
                    data_b = tlx.local_view(b, buf + NUM_STAGES)
                    tlx.async_descriptor_load(
                        b_desc,
                        data_b,
                        [offset_k, offset_bn],
                        full_b)

                    # Flip phase after every NUM_STAGES iterations finish
                    p = p ^ (buf == (NUM_STAGES-1))
                

                empty_a_2nd = tlx.local_view(bars_empty_a2, 1)
                empty_b_2 = tlx.local_view(bars_empty_b2, 1)
                last_dot_phase = 1
                tlx.barrier_wait(bar=empty_a_2nd, phase=last_dot_phase)
                tlx.barrier_wait(bar=empty_b_2, phase=last_dot_phase)

        # consumers (wgmma + async store)
        # first half of the tile A
        with tlx.async_task(num_warps=4):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            k_tiles = tl.cdiv(K, BK)
            num_tiles = num_pid_m * num_pid_n

            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
                p = 0
                acc = tl.zeros([BM//2, BN], dtype=tl.float32)
                for k in range(0, k_tiles):
                    buf = k % NUM_STAGES

                    # Wait for TMA load
                    full_a = tlx.local_view(bars_full_a1, buf) # noqa
                    full_b = tlx.local_view(bars_full_b1, buf)
                    tlx.barrier_wait(bar=full_a, phase=p)
                    tlx.barrier_wait(bar=full_b, phase=p)

                    # async_dot
                    data_a = tlx.local_view(a, buf) # noqa
                    data_b = tlx.local_view(b, buf)
                    acc = tlx.async_dot(
                        data_a,
                        data_b,
                        acc,
                    )
                    # async_wait
                    acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                    # Release buffers
                    empty_a = tlx.local_view(bars_empty_a1, buf) # noqa
                    empty_b = tlx.local_view(bars_empty_b1, buf)
                    tlx.barrier_arrive(empty_a)  # EmptyBar A1 arrive
                    tlx.barrier_arrive(empty_b)

                    # Flip phase after every NUM_STAGES iterations finish
                    p = p ^ (buf == (NUM_STAGES-1))
                
                act = gelu(acc)
                res = act.to(tlx.dtype_of(c_desc))
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_bn_c = pid_n * BN
                offs_am_c = pid_m * BM
                offset_cm = offs_am_c
                c_desc.store([offset_cm, offs_bn_c], res)  # noqa

        # consumers (wgmma + async store)
        # second half of the tile A
        with tlx.async_task(num_warps=4):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            k_tiles = tl.cdiv(K, BK)
            num_tiles = num_pid_m * num_pid_n

            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
                p = 0
                acc = tl.zeros([BM//2, BN], dtype=tl.float32)
                for k in range(0, k_tiles):
                    buf = k % NUM_STAGES

                    # Wait for TMA load
                    full_a = tlx.local_view(bars_full_a2, buf) # noqa
                    full_b = tlx.local_view(bars_full_b2, buf)
                    tlx.barrier_wait(bar=full_a, phase=p)
                    tlx.barrier_wait(bar=full_b, phase=p)

                    # async_dot
                    data_a = tlx.local_view(a, buf + NUM_STAGES) # noqa
                    data_b = tlx.local_view(b, buf + NUM_STAGES)
                    acc = tlx.async_dot(
                        data_a,
                        data_b,
                        acc,
                    )
                    # async_wait
                    acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                    # Release buffers
                    empty_a = tlx.local_view(bars_empty_a2, buf) # noqa
                    empty_b = tlx.local_view(bars_empty_b2, buf)
                    tlx.barrier_arrive(empty_a)  # EmptyBar A1 arrive
                    tlx.barrier_arrive(empty_b)

                    # Flip phase after every NUM_STAGES iterations finish
                    p = p ^ (buf == (NUM_STAGES-1))
                
                act = gelu(acc)
                res = act.to(tlx.dtype_of(c_desc))
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_bn_c = pid_n * BN
                offs_am_c = pid_m * BM
                offset_cm = offs_am_c + BLOCK_M_SPLIT
                c_desc.store([offset_cm, offs_bn_c], res)  # noqa


def tlx_ws_pingpong_v2(a, b,):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Illegal dimensions of input operands"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    (M, N, K) = (a.shape[0], b.shape[1], a.shape[1])
    c = torch.zeros((M, N), dtype=torch.float16, device=DEVICE, )

    dummy_block = [1, 1]
    desc_in_1 = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    desc_in_2 = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    desc_out = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"])), )

    tlx_ws_pingpong_v2_kernel[grid](
        desc_in_1, desc_in_2, desc_out,  #
        M, N, K,  #
        NUM_SMS=NUM_SMS,
    )
    return c

output_tlx_ws_pingpong_v2 = tlx_ws_pingpong_v2(a, b)
if torch.allclose(output_tlx_ws_pingpong_v2, output_ref, atol=1e-2, rtol=rtol):
    print("✅ TLX-WS-pingpong-V2 and Torch match")
else:
    print("❌ TLX-WS-pingpong-V2 and Torch differ")

ms = do_bench(lambda: tlx_ws_pingpong_v2(a, b))
# Calculate throughput metrics
tflops = flops / (ms * 1e-3) / 1e12  # TFLOPS
print(f"TLX-WS-pingpong-V2 Time: {ms:.3f} ms")
print(f"TLX-WS-pingpong-V2 TFLOPS: {tflops:.2f}")

#  =============== TLX Persistent WS PingPongV3 version of the GEMM+GELU kernel ===============

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BM": 128,
                "BN": 128,
                "BK": 64,
                "GROUP_SIZE_M": 8,
                "NUM_STAGES": 4, # must be even for now
                "NUM_MMA_WARPS": 8, # fixed
                "NUM_MMA_GROUPS": 2, # fixed
                "EPILOGUE_SUBTILE": False,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=matmul_tma_set_block_size_hook
        ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@triton.jit
def tlx_ws_pingpong_v3_kernel(
    a_desc, b_desc, c_desc,  #
    M, N, K,  #
    BM: tl.constexpr,  #
    BN: tl.constexpr,  #
    BK: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,  #
    NUM_MMA_WARPS: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
):
    # Descriptor
    BLOCK_M_SPLIT: tl.constexpr = BM // NUM_MMA_GROUPS

    a = tlx.local_alloc((BLOCK_M_SPLIT, BK), tlx.dtype_of(a_desc), NUM_STAGES * NUM_MMA_GROUPS)
    b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)

    bars_empty_a1 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_full_a1 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_empty_a2 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_full_a2 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_empty_b1 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_full_b1 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_empty_b2 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_full_b2 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    gemm_bars_a1 = tlx.alloc_barriers(num_barriers=1, arrive_count=2)
    gemm_bars_a2 = tlx.alloc_barriers(num_barriers=1, arrive_count=2)

    # Warp specilization
    with tlx.async_tasks():
        # Producer (async load)
        with tlx.async_task("default"):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            k_tiles = tl.cdiv(K, BK)
            num_tiles = num_pid_m * num_pid_n

            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False):
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)

                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                p = 1
                for k in range(0, k_tiles):
                    buf = k % NUM_STAGES
                    offset_k = k * BK

                    # Async load to a[buf]
                    empty_a_1st = tlx.local_view(bars_empty_a1, buf)  # mbar
                    full_a_1st = tlx.local_view(bars_full_a1, buf)  # mbar
                    tlx.barrier_wait(bar=empty_a_1st, phase=p)  # EmptyBar A1 wait
                    tlx.barrier_expect_bytes(full_a_1st, BLOCK_M_SPLIT * BK * 2)
                    data_a_1st = tlx.local_view(a, buf)  # smem data
                    tlx.async_descriptor_load(
                        a_desc,
                        data_a_1st,
                        [offset_am, offset_k],
                        full_a_1st)

                    # Async load to b[buf]
                    empty_b = tlx.local_view(bars_empty_b1, buf)
                    full_b = tlx.local_view(bars_full_b1, buf)
                    tlx.barrier_wait(bar=empty_b, phase=p)
                    tlx.barrier_expect_bytes(full_b, BN * BK * 2)
                    data_b = tlx.local_view(b, buf)
                    tlx.async_descriptor_load(
                        b_desc,
                        data_b,
                        [offset_k, offset_bn],
                        full_b)
                    # Flip phase after every NUM_STAGES iterations finish
                    p = p ^ (buf == (NUM_STAGES-1))
                empty_a_1st = tlx.local_view(bars_empty_a1, 1)  # mbar
                empty_b_1 = tlx.local_view(bars_empty_b1, 1)
                last_dot_phase = 1
                tlx.barrier_wait(bar=empty_a_1st, phase=last_dot_phase)  # EmptyBar A1 wait
                tlx.barrier_wait(bar=empty_b_1, phase=last_dot_phase)

                p = 1
                for k in range(0, k_tiles):
                    buf = k % NUM_STAGES
                    offset_k = k * BK

                    # Async load to a[buf+NUM_STAGES]
                    empty_a_2nd = tlx.local_view(bars_empty_a2, buf)
                    full_a_2nd = tlx.local_view(bars_full_a2, buf)
                    tlx.barrier_wait(bar=empty_a_2nd, phase=p)
                    tlx.barrier_expect_bytes(bar=full_a_2nd, size=BLOCK_M_SPLIT * BK * 2)
                    data_a_2nd = tlx.local_view(a, buf + NUM_STAGES)  # smem data
                    tlx.async_descriptor_load(
                        a_desc,
                        data_a_2nd,
                        [offset_am + BLOCK_M_SPLIT, offset_k],
                        full_a_2nd)

                    # Async load to b[buf]
                    empty_b = tlx.local_view(bars_empty_b2, buf)
                    full_b = tlx.local_view(bars_full_b2, buf)
                    tlx.barrier_wait(bar=empty_b, phase=p)
                    tlx.barrier_expect_bytes(full_b, BN * BK * 2)
                    data_b = tlx.local_view(b, buf)
                    tlx.async_descriptor_load(
                        b_desc,
                        data_b,
                        [offset_k, offset_bn],
                        full_b)

                    # Flip phase after every NUM_STAGES iterations finish
                    p = p ^ (buf == (NUM_STAGES-1))
                
                empty_a_2nd = tlx.local_view(bars_empty_a2, 1)
                empty_b_2 = tlx.local_view(bars_empty_b2, 1)
                last_dot_phase = 1
                tlx.barrier_wait(bar=empty_a_2nd, phase=last_dot_phase)
                tlx.barrier_wait(bar=empty_b_2, phase=last_dot_phase)  

        # consumers (wgmma + async store)
        # first half of the tile A
        with tlx.async_task(num_warps=4):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            k_tiles = tl.cdiv(K, BK)
            num_tiles = num_pid_m * num_pid_n

            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            p_gemm = 0
            for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False):
                p = 0
                acc = tl.zeros([BM//2, BN], dtype=tl.float32)
                for k in range(0, k_tiles):
                    buf = k % NUM_STAGES

                    # Wait for TMA load
                    full_a = tlx.local_view(bars_full_a1, buf) # noqa
                    full_b = tlx.local_view(bars_full_b1, buf)
                    tlx.barrier_wait(bar=full_a, phase=p)
                    tlx.barrier_wait(bar=full_b, phase=p)

                    # async_dot
                    data_a = tlx.local_view(a, buf) # noqa
                    data_b = tlx.local_view(b, buf)
                    acc = tlx.async_dot(
                        data_a,
                        data_b,
                        acc,
                    )
                    # async_wait
                    acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                    # Release buffers
                    empty_a = tlx.local_view(bars_empty_a1, buf) # noqa
                    empty_b = tlx.local_view(bars_empty_b1, buf)
                    tlx.barrier_arrive(empty_a)  # EmptyBar A1 arrive
                    tlx.barrier_arrive(empty_b)

                    # Flip phase after every NUM_STAGES iterations finish
                    p = p ^ (buf == (NUM_STAGES-1))

                gemm_bar1 = tlx.local_view(gemm_bars_a1, 0)
                tlx.barrier_arrive(gemm_bar1)
                tlx.barrier_wait(bar=gemm_bar1, phase=p_gemm)

                act = gelu(acc)
                res = act.to(tlx.dtype_of(c_desc))
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_bn_c = pid_n * BN
                offs_am_c = pid_m * BM
                offset_cm = offs_am_c
                c_desc.store([offset_cm, offs_bn_c], res)  # noqa

                gemm_bar2 = tlx.local_view(gemm_bars_a2, 0)
                tlx.barrier_arrive(gemm_bar2)
                tlx.barrier_wait(bar=gemm_bar2, phase=p_gemm)

                p_gemm = p_gemm ^ 1


        # consumers (wgmma + async store)
        # second half of the tile A
        with tlx.async_task(num_warps=4):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            k_tiles = tl.cdiv(K, BK)
            num_tiles = num_pid_m * num_pid_n

            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            p_gemm = 0
            gemm_bar1 = tlx.local_view(gemm_bars_a1, 0)
            tlx.barrier_arrive(gemm_bar1)
            tlx.barrier_wait(bar=gemm_bar1, phase=p_gemm)
            for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False):
                p = 0
                acc = tl.zeros([BM//2, BN], dtype=tl.float32)
                for k in range(0, k_tiles):
                    buf = k % NUM_STAGES

                    # Wait for TMA load
                    full_a = tlx.local_view(bars_full_a2, buf) # noqa
                    full_b = tlx.local_view(bars_full_b2, buf)
                    tlx.barrier_wait(bar=full_a, phase=p)
                    tlx.barrier_wait(bar=full_b, phase=p)

                    # async_dot
                    data_a = tlx.local_view(a, buf + NUM_STAGES) # noqa
                    data_b = tlx.local_view(b, buf)
                    acc = tlx.async_dot(
                        data_a,
                        data_b,
                        acc,
                    )
                    # async_wait
                    acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                    # Release buffers
                    empty_a = tlx.local_view(bars_empty_a2, buf) # noqa
                    empty_b = tlx.local_view(bars_empty_b2, buf)
                    tlx.barrier_arrive(empty_a)  # EmptyBar A1 arrive
                    tlx.barrier_arrive(empty_b)

                    # Flip phase after every NUM_STAGES iterations finish
                    p = p ^ (buf == (NUM_STAGES-1))

                gemm_bar2 = tlx.local_view(gemm_bars_a2, 0)
                tlx.barrier_arrive(gemm_bar2)
                tlx.barrier_wait(bar=gemm_bar2, phase=p_gemm)

                act = gelu(acc)

                res = act.to(tlx.dtype_of(c_desc))
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_bn_c = pid_n * BN
                offs_am_c = pid_m * BM
                offset_cm = offs_am_c + BLOCK_M_SPLIT
                c_desc.store([offset_cm, offs_bn_c], res)  # noqa

                if tile_id < num_tiles - NUM_SMS:
                    gemm_bar1 = tlx.local_view(gemm_bars_a1, 0)
                    tlx.barrier_arrive(gemm_bar1)
                    tlx.barrier_wait(bar=gemm_bar1, phase=p_gemm ^ 1)

                p_gemm = p_gemm ^ 1


def tlx_ws_pingpong_v3(a, b,):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Illegal dimensions of input operands"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    (M, N, K) = (a.shape[0], b.shape[1], a.shape[1])
    c = torch.zeros((M, N), dtype=torch.float16, device=DEVICE, )

    dummy_block = [1, 1]
    desc_in_1 = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    desc_in_2 = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    desc_out = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"])), )

    tlx_ws_pingpong_v3_kernel[grid](
        desc_in_1, desc_in_2, desc_out,  #
        M, N, K,  #
        NUM_SMS=NUM_SMS,
    )
    return c

output_tlx_ws_pingpong_v3 = tlx_ws_pingpong_v3(a, b)
if torch.allclose(output_tlx_ws_pingpong_v3, output_ref, atol=1e-2, rtol=rtol):
    print("✅ TLX-WS-pingpong-V3 and Torch match")
else:
    print("❌ TLX-WS-pingpong-V3 and Torch differ")

ms = do_bench(lambda: tlx_ws_pingpong_v3(a, b))
# Calculate throughput metrics
tflops = flops / (ms * 1e-3) / 1e12  # TFLOPS
print(f"TLX-WS-pingpong-V3 Time: {ms:.3f} ms")
print(f"TLX-WS-pingpong-V3 TFLOPS: {tflops:.2f}")