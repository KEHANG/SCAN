#!/usr/bin/env python
# coding=utf-8

import random
import torch

a = torch.zeros([10, 10], dtype=torch.uint8)
x = 10
y = 10
ratio = 0.3
x_span = int(x * ratio)
y_span = int(y * ratio)
left = random.randint(0, x - x_span)
down = random.randint(0, y - y_span)
print(a.sum())
a[left : left + x_span, down : down + y_span] = 1
print(a.sum())
