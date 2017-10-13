from __future__ import print_function

import torch

x = torch.Tensor(5, 3)
print(x)

y = torch.rand(5, 3)
print(x + y)

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    z = x + y
    print(z)
else:
    print("Sorry, CUDA not available")