import torch
print(torch.__version__)          # 看版本号
print(torch.version.cuda)         # 看是否绑定 CUDA
print(torch.backends.cudnn.enabled)  # 看 CUDNN 是否可用