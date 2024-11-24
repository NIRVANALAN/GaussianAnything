"""
Helpers for distributed training.
"""

import datetime
import io
import os
import socket

import blobfile as bf
from pdb import set_trace as st
# from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8
SETUP_RETRY_COUNT = 3


def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()


def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def setup_dist(args):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    # print(f"{os.environ['MASTER_ADDR']=} {args.master_port=}")

    # dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=th.cuda.device_count(),  timeout=datetime.timedelta(seconds=5400))
    # st() no mark
    dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=54000))
    print(f"{args.local_rank=} init complete")

    # synchronize() # extra memory on rank 0, why?

    th.cuda.empty_cache()

def cleanup():
    dist.destroy_process_group()

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():

        if get_world_size() > 1:
            return th.device(f"cuda:{get_rank() % GPUS_PER_NODE}")
        return th.device(f"cuda")

    return th.device("cpu")


# def load_state_dict(path, submodule_name='', **kwargs):
def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    # chunk_size = 2 ** 30  # MPI has a relatively small size limit
    # if get_rank() == 0:
    #     with bf.BlobFile(path, "rb") as f:
    #         data = f.read()
    #     num_chunks = len(data) // chunk_size
    #     if len(data) % chunk_size:
    #         num_chunks += 1
    #     MPI.COMM_WORLD.bcast(num_chunks)
    #     for i in range(0, len(data), chunk_size):
    #         MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    # else:
    #     num_chunks = MPI.COMM_WORLD.bcast(None)
    #     data = bytes()
    #     for _ in range(num_chunks):
    #         data += MPI.COMM_WORLD.bcast(None)

    # return th.load(io.BytesIO(data), **kwargs)
    # with open(path) as f:
    ckpt = th.load(path, **kwargs)
    # if submodule_name != '':
    #     assert submodule_name in ckpt
    #     return ckpt[submodule_name]
    # else:
    return ckpt


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    # for k, p in params:
    for p in params:
        with th.no_grad():
            try:
                dist.broadcast(p, 0)
            except Exception as e:
                print(k, e)
                # print(e)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


_num_moments    = 3             # [num_scalars, sum_of_scalars, sum_of_squares]
_reduce_dtype   = th.float32 # Data type to use for initial per-tensor reduction.
_counter_dtype  = th.float64 # Data type to use for the internal counters.
_rank           = 0             # Rank of the current process.
_sync_device    = None          # Device to use for multiprocess communication. None = single-process.
_sync_called    = False         # Has _sync() been called yet?
_counters       = dict()        # Running counters on each device, updated by report(): name => device => torch.Tensor
_cumulative     = dict()        # Cumulative counters on the CPU, updated by _sync(): name => torch.Tensor

def init_multiprocessing(rank, sync_device):
    r"""Initializes `torch_utils.training_stats` for collecting statistics
    across multiple processes.
    This function must be called after
    `torch.distributed.init_process_group()` and before `Collector.update()`.
    The call is not necessary if multi-process collection is not needed.
    Args:
        rank:           Rank of the current process.
        sync_device:    PyTorch device to use for inter-process
                        communication, or None to disable multi-process
                        collection. Typically `torch.device('cuda', rank)`.
    """
    global _rank, _sync_device
    assert not _sync_called
    _rank = rank
    _sync_device = sync_device