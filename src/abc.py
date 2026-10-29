import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.device_name(0))  # Check the GPU name
