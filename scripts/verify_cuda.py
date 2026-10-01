import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    # Tiny tensor operation to prove GPU execution
    try:
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        y = torch.tensor([4.0, 5.0, 6.0]).cuda()
        z = x + y
        print(f"GPU Tensor Addition Result: {z}")
        print("CUDA Execution Verified Successfully.")
    except Exception as e:
        print(f"CUDA Execution Failed: {e}")
else:
    print("CUDA is NOT available. Falling back to CPU.")
