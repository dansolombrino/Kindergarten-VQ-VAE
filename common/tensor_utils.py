from rich import print

import torch

from torch import Tensor
from torch import IntTensor

import math

def replace_pct_rand_values(
    tensor: IntTensor, percentage: float, 
    rand_int_low: int, rand_int_high: int
):
    
    if math.isclose(percentage, 0):
        return tensor
    
    device = tensor.get_device()

    # print(f"tensor:\n{tensor}")

    tot_num_els = tensor.numel()
    
    # print(f"tot_num_els: {tot_num_els}")

    num_zero_els = int(tot_num_els * percentage)
    # print(f"num_zero_els: {num_zero_els}")
    num_one_els  = tot_num_els - num_zero_els
    # print(f"num_one_els: {num_one_els}")

    mask_to_keep = torch.cat((torch.zeros(num_zero_els), torch.ones(num_one_els)))

    mask_to_keep = mask_to_keep[torch.randperm(mask_to_keep.size(0))]

    mask_to_keep = mask_to_keep.reshape(tensor.shape).bool().to(device)
    
    # print(f"mask_to_keep:\n{mask_to_keep}")

    noise = torch.randint_like(tensor, rand_int_low, rand_int_high).to(device)
    # print(f"noise:\n{noise}")

    corrupted = torch.where(mask_to_keep, tensor, noise)

    # print(f"corrupted:\n{corrupted}")

    return corrupted






def main():
    rand_int_low = 0
    rand_int_high = 10
    t = torch.randint(rand_int_low, rand_int_high, (2, 10))

    replace_pct_rand_values(t, 0.2, rand_int_low, rand_int_high)

if __name__ == "__main__":

    main()
