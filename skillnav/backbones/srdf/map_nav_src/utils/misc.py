import random
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# from .logger import LOGGER

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
    return mask

# def wrap_model(
#     model: torch.nn.Module, device: torch.device, local_rank: int
# ) -> torch.nn.Module:
#     model.to(device)

#     if local_rank != -1:
#         model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
#         # At the time of DDP wrapping, parameters and buffers (i.e., model.state_dict()) 
#         # on rank0 are broadcasted to all other ranks.
#     elif torch.cuda.device_count() > 1:
#         LOGGER.info("Using data parallel")
#         model = torch.nn.DataParallel(model)

#     return model