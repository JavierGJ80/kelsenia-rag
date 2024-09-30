import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. GPU will be used for computations.")
else:
    print("CUDA is not available. Computations will be on CPU.")