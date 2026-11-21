
import torch
import torchvision

print("=" * 50)
print("RETINAL DISEASE DETECTION - GPU TEST")
print("=" * 50)

print("PyTorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)
print("PyTorch CUDA Version:", torch.version.cuda)

if torch.cuda.is_available():
    device = torch.device("cuda")

    print("CUDA Available: YES")
    print("GPU:", torch.cuda.get_device_name(0))

    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    print("GPU Memory:", round(gpu_memory / 1024**3, 2), "GB")

    # Test a small GPU computation
    x = torch.randn(100, 100, device=device)
    y = torch.matmul(x, x)

    print("GPU Tensor Device:", y.device)
    print("GPU Computation: SUCCESS")

else:
    print("CUDA Available: NO")
    print("PyTorch is using CPU.")

print("=" * 50)
