#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 29 09:27:22 2022

@author: arek

https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d
https://stackoverflow.com/questions/58692476/what-is-adaptive-average-pooling-and-how-does-it-work
"""

m = nn.AdaptiveAvgPool2d((2,2))
input = torch.randint(10, (1, 3, 4, 4)).type(torch.float)
input
input.shape     # torch.Size([1, 3, 4, 4])
output = m(input)
output
output.shape    # torch.Size([1, 3, 2, 2])
output.view(1, 12)
output.view(1, -1)

# target output size of 5x7
m = nn.AdaptiveAvgPool2d((5,7))
input = torch.randn(1, 64, 8, 9)
input.shape     # torch.Size([1, 64, 8, 9])
output = m(input)
output.shape    # torch.Size([1, 64, 5, 7])

# target output size of 7x7 (square)
m = nn.AdaptiveAvgPool2d(7)
input = torch.randn(1, 64, 10, 9)
input.shape     # torch.Size([1, 64, 8, 9])
output = m(input)
output.shape    # torch.Size([1, 64, 7, 7])

# target output size of 10x7
m = nn.AdaptiveAvgPool2d((None, 7))
input = torch.randn(1, 64, 10, 9)
output = m(input)
output.shape    # torch.Size([1, 64, 10, 7])
