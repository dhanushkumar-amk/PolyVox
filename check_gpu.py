import torch


def main():
    print("--- PyTorch Hardware Check ---")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"GPU Count: {device_count}")
        print(f"Active GPU: {device_name}")
        device = torch.device("cuda")
    else:
        print("Running in CPU mode")
        device = torch.device("cpu")

    # Run dummy tensor math to verify functionality
    print("\nRunning dummy tensor calculation on device...")
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)
    result = x * y + 2.0

    print(f"Tensor computation successful on [{device}]: {result.tolist()}")
    print("PyTorch is working correctly!")


if __name__ == "__main__":
    main()
