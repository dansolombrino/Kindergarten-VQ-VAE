from rich import print


from torch import Tensor

import torch

def seq_acc(input: Tensor, target: Tensor) -> float:

    assert input.shape == target.shape, "input and target shapes must match"

    assert not input.is_floating_point(), "input tensor must be integer type, not floating point"
    assert not target.is_floating_point(), "target tensor must be integer type, not floating point"

    # print(f"input: {input}\n")
    # print(f"target: {target}\n")

    diff: Tensor = input - target

    # print(f"diff: {diff}\n")

    mask = diff == 0

    # print(f"mask: {mask}\n")

    sum = mask.sum()

    # print(f"sum: {sum}\n")

    acc = sum / input.numel()

    # print(f"acc: {acc}\n")
    

    return acc


def main():

    input  = torch.randint(10, 12, (3, 5))
    target = torch.randint(10, 12, (3, 5))

    seq_acc(input, target)

if __name__ == "__main__":

    main()