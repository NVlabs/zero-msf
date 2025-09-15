# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/


import torch

import zmsf.utils.path_to_mast3r
import mast3r.utils.path_to_dust3r
from dust3r.losses import Sum, BaseCriterion, LLoss, L21Loss, Criterion, MultiLoss


L21 = L21Loss()

class L11Loss (LLoss):
    def distance(self, a, b):
        return torch.sum(torch.abs(a - b), dim=-1)

L11 = L11Loss()
