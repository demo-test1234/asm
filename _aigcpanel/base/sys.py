import torch


def cudaIsEnable():
    try:
        return torch.cuda.is_available()
    except:
        return False


def cudaGpuSize():
    try:
        device = torch.device('cuda')
        gpu_properties = torch.cuda.get_device_properties(device)
        total_memory = gpu_properties.total_memory
        total_memory_gb = total_memory / (1024 ** 3)
        return total_memory_gb
    except:
        return 0
