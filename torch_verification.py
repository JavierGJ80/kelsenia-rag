import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. GPU will be used for computations.")
else:
    print("CUDA is not available. Computations will be on CPU.")

# Check if MPS is available
if torch.backends.mps.is_available():
    print("MPS backend is available")
else:
    print("MPS backend is not available")