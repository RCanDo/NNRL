#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:22:35 2022

@author: arek
"""
#%%
import torch
import torch.nn as nn


#%%

# With Learnable Parameters
m = nn.BatchNorm2d(100)

# Without Learnable Parameters
m = nn.BatchNorm2d(100, affine=False)
input = torch.randn(20, 100, 35, 45)
output = m(input)
output.shape    # torch.Size([20, 100, 35, 45])

m.parameters

for p in m.parameters(): print(p)

dir(m)
m.named_parameters

#%%



#%%



#%%
