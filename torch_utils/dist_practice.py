import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def setup_process(rank, master_addr, master_port, world_size, backend='nccl'):
    print(f'setting up {rank=} {world_size=} {backend=}')

    # set up the master's ip address so this child process can coordinate
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    print(f"{master_addr=} {master_port=}")

    # Initializes the default distributed process group, and this will also initialize the distributed package.
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"{rank=} init complete")
    dist.destroy_process_group()
    print(f"{rank=} destroy complete")


if __name__ == '__main__':
    world_size = 2
    master_addr = '127.0.0.1'
    master_port = find_free_port()
    mp.spawn(setup_process,
             args=(
                 master_addr,
                 master_port,
                 world_size,
             ),
             nprocs=world_size)
