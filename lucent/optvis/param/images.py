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

"""High-level wrapper for paramaterizing images."""

from __future__ import absolute_import, division, print_function

from typing import Optional, Tuple, Callable, List

import torch

from lucent.optvis.param.spatial import pixel_image, fft_image
from lucent.optvis.param.color import to_valid_rgb

def image(
    w: int, 
    h: Optional[int] = None, 
    sd: Optional[float] = 0.01, 
    batch: Optional[int] = 1, 
    decorrelate: Optional[bool] = True,
    fft: Optional[bool] = True, 
    channels: Optional[int] = None,
) -> Tuple[List[torch.Tensor], Callable]:
    """Creates image parameterization either via rgb or fft.

    :param w: image width
    :type w: int
    :param h: image height, if None will be the same as ``w``, defaults to None
    :type h: Optional[int], optional
    :param sd: standard deviation of random init, defaults to 0.01
    :type sd: Optional[float], optional
    :param batch: [description], defaults to None
    :type batch: Optional[int], optional, defaults to 1.
    :param decorrelate: [description], defaults to True
    :type decorrelate: Optional[bool], optional
    :param fft: [description], defaults to True
    :type fft: Optional[bool], optional
    :param channels: number of channels - if None will be set to 3, defaults to None
    :type channels: Optional[int], optional
    :return: [description]
    :rtype: Tuple[List[torch.Tensor], Callable]
    """

    # set default args
    if h is None:
        h = w
    if channels is None:
        ch = 3
    else:
        ch = channels

    shape = [batch, ch, h, w]
    param_f = fft_image if fft else pixel_image
    params, image_f = param_f(shape, sd=sd)
    if channels:
        output = to_valid_rgb(image_f, decorrelate=False) #TODO does this make sense?
    else:
        output = to_valid_rgb(image_f, decorrelate=decorrelate)
    return params, output
