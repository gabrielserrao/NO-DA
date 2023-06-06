import torch

print ("Is GPU available: ", torch.cuda.is_available())
print ("Number of GPUs available: ", torch.cuda.device_count())
