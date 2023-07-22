import torch


def main():
    # load the library, assume it is located together with this file
    torch.ops.load_library("../libIntelliStream.so")
    # gen the input tensor
    a = torch.rand(100, 100)
    b = torch.rand(100, 100)
    # The pytorch +
    print('/****test add****/')
    print('pytorch-mm:', torch.matmul(a, b))
    # our c++ extension of +
    torch.ops.load_library("../libIntelliStream.so")
    torch.ops.AMMBench.setTag('mm')
    print('AMMBench-MM+:', torch.ops.AMMBench.ammDefault(a, b))
    torch.ops.AMMBench.setTag('crs')
    print('AMMBench-CRS+:', torch.ops.AMMBench.ammDefault(a, b))


if __name__ == "__main__":
    main()
