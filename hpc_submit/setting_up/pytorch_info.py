#!/usr/bin/env python
"""Show pytorch info."""

import torch

print(f"pytorch version: {torch.__version__}")

dev_cnt = torch.cuda.device_count()
print(f"Found {dev_cnt} devices!")
for i in range(dev_cnt):
   dev = torch.cuda.get_device_properties(i)
   print(dev.name)
   print(dev)
   print()
