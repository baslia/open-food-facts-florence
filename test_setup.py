#%%
import torch

print(torch.cuda.is_available())

print(torch.__version__)
print(f"MPS available: {torch.backends.mps.is_available()}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")