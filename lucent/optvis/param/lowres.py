# Copyright 2020 The Lucent Authors. All Rights Reserved.
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

"""Provides lowres_tensor()."""

from __future__ import absolute_import, division, print_function

from typing import Union, List, Tuple, Optional, Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F

def lowres_tensor(
    shape: Union[List, Tuple, torch.Size], 
    underlying_shape: Union[List, Tuple, torch.Size], 
    offset: Optional[Union[bool, int, List]] = None, 
    sd: Optional[float] = 0.01,
) -> Tuple[List[torch.Tensor], Callable]:
    """Produces a tensor paramaterized by a interpolated lower resolution tensor.
    This is like what is done in a laplacian pyramid, but a bit more general. It
    can be a powerful way to describe images.

    :param shape: desired shape of resulting tensor, should be of format (B, C, H, W) #TODO support more shapes
    :type shape: Union[List, Tuple, torch.Size]
    :param underlying_shape: shape of the tensor being resized into final tensor
    :type underlying_shape: Union[List, Tuple, torch.Size]
    :param offset: Describes how to offset the interpolated vector (like phase in a
            Fourier transform). If None, apply no offset. If int, apply the same
            offset to each dimension; if a list use each entry for each dimension.
            If False, do not offset. If True, offset by half the ratio between shape and underlying shape (analogous to 90
            degrees), defaults to None
    :type offset: Optional[Union[bool, int, List]], optional
    :param sd: Standard deviation of initial tensor variable., defaults to 0.01
    :type sd: Optional[float], optional
    :return: One-element list containing the low resolution tensor and the corresponding image function returning the tensor on call.
    :rtype: Tuple[List[torch.Tensor], Callable]
    """
    if isinstance(offset, float):
        raise TypeError('Passing float offset is deprecated!')

    # TODO pass device as argument to avoid mixing devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    underlying_t = (torch.randn(*underlying_shape) * sd).to(device).requires_grad_(True)
    if offset is not None:
        # Deal with non-list offset
        if not isinstance(offset, list):
            offset = len(shape) * [offset]
        # Deal with the non-int offset entries
        for n in range(len(offset)):
            if offset[n] is True:
                offset[n] = shape[n] / underlying_shape[n] / 2
            if offset[n] is False:
                offset[n] = 0
            offset[n] = int(offset[n])
    
    underlying_t = einops.rearrange(underlying_t, 'b c h w -> c b h w')

    def inner():
        t = torch.nn.functional.interpolate(underlying_t[None], (shape[0], shape[2], shape[3]), mode="trilinear")
        t = einops.rearrange(t, 'dummy c b h w -> (dummy b) c h w')
        if offset is not None:
            # Actually apply offset by padding and then cropping off the excess.
            t = F.pad(t, offset, "reflect")
            t = t[:shape[0], :shape[1], :shape[2], :shape[3]]
        return t
    return [underlying_t], inner