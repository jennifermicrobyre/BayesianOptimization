
import torch
from botorch.utils.transforms import unnormalize, normalize

# !!!!!!!!
# If all elements of X are contained within bounds, the normalized values will be contained within [0, 1]^d
# !!!!!!!!

X = 2 * torch.rand(4, 3)
print(X)
bounds = torch.stack([torch.zeros(3), 2 * torch.ones(3)])
print(bounds)
X_normalized = normalize(X, bounds)
print(X_normalized)
X = unnormalize(X_normalized, bounds)
print(X)
