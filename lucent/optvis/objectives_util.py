# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions for Objectives."""

from __future__ import absolute_import, division, print_function

from typing import Optional, Any

import torch


def _make_arg_str(arg: Any) -> str:
    """Helper function to convert arg to str.

    :param arg: argument
    :type arg: Any
    :return: arg converted to str
    :rtype: str
    """
    arg = str(arg)
    too_big = len(arg) > 15 or "\n" in arg
    return "..." if too_big else arg


def _extract_act_pos(
    acts: torch.Tensor, 
    x: Optional[int] = None, 
    y: Optional[int] = None
) -> torch.Tensor:
    """Given a tensor of activations, returns the part of the activations that correspond to the given (x,y) position.

    :param acts: Activation tensor
    :type acts: torch.Tensor
    :param x: x-coordinate, defaults to None
    :type x: Optional[int], optional
    :param y: y-coordinate, defaults to None
    :type y: Optional[int], optional
    :return: activation at coordinate (x, y) 
    :rtype: torch.Tensor
    """
    shape = acts.shape
    if len(shape) not in (3, 4):
        raise ValueError(f"Expected activations to be of shape (B C H W) or (C H W) but got shape {shape}")
        
    x = shape[2] // 2 if x is None else x #TODO move this default case outside of this function?
    y = shape[3] // 2 if y is None else y
    return acts[..., y:y+1, x:x+1]


def _T_handle_batch(T, batch=None):
    def T2(name):
        t = T(name)
        if isinstance(batch, int):
            return t[batch:batch+1]
        else:
            return t
    return T2
