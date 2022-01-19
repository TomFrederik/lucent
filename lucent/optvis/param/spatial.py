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

from __future__ import absolute_import, division, print_function

from typing import Union, List, Tuple, Optional, Callable

import numpy as np
import torch

# TODO pass this as an argument to avoid mixing devices 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TORCH_VERSION = torch.__version__


def pixel_image(
    shape: Union[List, Tuple, torch.Size], 
    sd: Optional[float] = 0.01,
) -> Tuple[List[torch.Tensor], Callable]:
    """Pixel parameterization of an image with given shape. 
    
    Init via zero-mean normal distribution with given standard deviation.

    :param shape: Shape of image
    :type shape: Union[List, Tuple, torch.Size]
    :param sd: Standard deviation of initializing distribution, defaults to 0.01
    :type sd: Optional[float], optional
    :return: one-element list containing image parameters and image function returning the image on call.
    :rtype: Tuple[List[torch.Tensor], Callable]
    """
    tensor = (torch.randn(*shape) * sd).to(device).requires_grad_(True)
    return [tensor], lambda: tensor



def rfft2d_freqs(
    h: int, 
    w: int,
) -> np.ndarray:
    """Computes 2D spectrum frequencies of an image with given dimensions.

    From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py .

    :param h: image height
    :type h: int
    :param w: image width
    :type w: int
    :return: 2D spectrum frequencies
    :rtype: np.ndarray
    """
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


def fft_image(
    shape: Union[List, Tuple, torch.Size], 
    sd: Optional[float] = 0.01, 
    decay_power: Optional[int] = 1,
) -> Tuple[List[torch.Tensor], Callable]:
    """Image parameterization via fft spectrum

    :param shape: Image shape
    :type shape: Union[List, Tuple, torch.Size]
    :param sd: Standard deviation for initializing distribution, defaults to 0.01
    :type sd: Optional[float], optional
    :param decay_power: Decay power to dampen scaling, defaults to 1
    :type decay_power: Optional[int], optional
    :return: One-element list containing image parameters and image function returning the image on call.
    :rtype: Tuple[List[Tensor], Callable]
    """
    batch, channels, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (batch, channels) + freqs.shape + (2,) # 2 for imaginary and real components

    spectrum_real_imag_t = (torch.randn(*init_val_size) * sd).to(device).requires_grad_(True)

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale = torch.tensor(scale).float()[None, None, ..., None].to(device)

    def inner():
        scaled_spectrum_t = scale * spectrum_real_imag_t
        if TORCH_VERSION >= "1.7.0":
            import torch.fft
            if type(scaled_spectrum_t) is not torch.complex64:
                scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
            image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm='ortho')
        else:
            import torch
            image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w))
        image = image[:batch, :channels, :h, :w]
        magic = 4.0 # Magic constant from Lucid library; increasing this seems to reduce saturation
        image = image / magic
        return image
    return [spectrum_real_imag_t], inner
