# import torch library
import torch
  

dec_out = torch.tensor([[[0.1, 0.7, 0.2],
                        [0.4, 0.5, 0.1],
                        [0.9, 0.1, 0.0]]])

# Find the maximum values along the last dimension
max_values, _ = torch.max(dec_out, dim=-1, keepdim=True)

# Create a mask by comparing each element to the maximum values
mask = torch.eq(dec_out, max_values)

# Convert the mask to the desired data type (float)
argmaxout = mask.float()

print(argmaxout)