from rich import print

import torch

from torch import Tensor
from torch import IntTensor

import math

import random


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


def change_percentage_of_elements(tensor: Tensor, dim, percentage, min, max):

    if math.isclose(percentage, 0): 
        return tensor
    
    # Calculate the number of elements to change
    num_elements = tensor.size(dim)
    num_to_change = int(num_elements * percentage)
    # print(f"num_to_change: {num_to_change}")

    # Generate random indices along the specified dimension
    indices = list(range(num_elements))
    random.shuffle(indices)
    indices = indices[:num_to_change]

    # Generate random values to replace the selected elements
    random_values = torch.randint(min, max, (1, num_to_change)).to(tensor.get_device())

    # Clone the original tensor to avoid modifying it in-place
    modified_tensor = tensor.clone()

    # Replace the selected elements with random values
    # if dim == 0:
    #     modified_tensor[indices, :, :] = random_values.view(-1, 1, 1)
    # elif dim == 1:
    #     modified_tensor[:, indices, :] = random_values.view(1, -1, 1)
    # elif dim == 2:
    #     modified_tensor[:, :, indices] = random_values.view(1, 1, -1)
    if dim == 0:
        modified_tensor[indices, :] = random_values.view(-1, 1)
    elif dim == 1:
        modified_tensor[:, indices] = random_values.view(1, -1)
    else:
        raise ValueError("Unsupported dimension")

    return modified_tensor


if __name__ == "__main__":
    tensor = torch.randn(20, 8)  # Example tensor
    percentage = 0.6  # Change 30% of elements
    dim = 1  # Along the third dimension

    print("Original Tensor:")
    print(tensor)

    modified_tensor = change_percentage_of_elements(tensor, dim, percentage)
    print("\nModified Tensor:")
    print(modified_tensor)

    print("\nDifferences (Modified - Original):")
    differences = modified_tensor - tensor
    print(differences)







# def main():
#     rand_int_low = 0
#     rand_int_high = 10
#     t = torch.randint(rand_int_low, rand_int_high, (2, 10))

#     replace_pct_rand_values(t, 0.2, rand_int_low, rand_int_high)

# if __name__ == "__main__":

#     main()
