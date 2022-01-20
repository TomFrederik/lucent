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

from typing import Callable, Optional, Iterable, Union, List, Tuple

import kornia
from kornia.geometry.transform import translate
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
KORNIA_VERSION = kornia.__version__


def jitter(d: int) -> Callable:
    """Returns a transform that randomly translates an image up to ``d`` pixels in either direction.

    :param d: Maximum absolute shift in x/y direction
    :type d: int
    :raises ValueError: d should be > 1
    :return: Transform function
    :rtype: Callable
    """
    if d <= 1:
        raise ValueError(f'jitter parameter d should be > 1 but is {d}')

    def inner(image_t):
        dx = np.random.choice(d)
        dy = np.random.choice(d)
        return translate(image_t, torch.tensor([[dx, dy]]).float().to(device))

    return inner


def pad(
    w: int, 
    mode: Optional[str] = "reflect", 
    constant_value: Optional[float] = 0.5,
) -> Callable:
    """Returns a transform that pads the input tensor.

    :param w: How many rows/columns to pad *on each* side
    :type w: int
    :param mode: padding mode - one of 'reflect', constant, 'replicate', or 'circular', defaults to "reflect"
    :type mode: Optional[str], optional
    :param constant_value: value that the image is padded with, defaults to 0.5
    :type constant_value: Optional[float], optional
    :return: Transform function
    :rtype: Callable
    """
    if mode != "constant":
        constant_value = 0

    def inner(image_t):
        return F.pad(image_t, [w] * 4, mode=mode, value=constant_value,)

    return inner


#TODO: not sure this is implemented correctly/most efficiently, why not just use interpolate?
def random_scale(scales: Union[List, Tuple, np.ndarray]) -> Callable:
    """Returns a transform that randomly scales the input tensor.

    :param scales: list, tuple, or ndarray of allowed scales
    :type scales: Union[List, Tuple, np.ndarray]
    :return: Transform function
    :rtype: Callable
    """
    def inner(image_t):
        scale = np.random.choice(scales)
        shp = image_t.shape[2:]
        scale_shape = [_roundup(scale * d) for d in shp]
        pad_x = max(0, _roundup((shp[1] - scale_shape[1]) / 2))
        pad_y = max(0, _roundup((shp[0] - scale_shape[0]) / 2))
        upsample = torch.nn.Upsample(
            size=scale_shape, mode="bilinear", align_corners=True
        )
        return F.pad(upsample(image_t), [pad_y, pad_x] * 2)

    return inner


def random_rotate(
    angles: Union[List, Tuple, np.ndarray], 
    units: Optional[str] = "degrees"
) -> Callable:
    """Returns a transform that randomly rotates an input image.

    :param angles: list, tuple, or np.ndarray of allowed angles.
    :type angles: Union[List, Tuple, np.ndarray]
    :param units: "degrees" or "radians", defaults to "degrees"
    :type units: Optional[str], optional
    :return: Transform function
    :rtype: Callable
    """
    def inner(image_t):
        b, _, h, w = image_t.shape
        # kornia takes degrees
        alpha = _rads2angle(np.random.choice(angles), units)
        angle = torch.ones(b) * alpha
        if KORNIA_VERSION < '0.4.0':
            scale = torch.ones(b)
        else:
            scale = torch.ones(b, 2)
        center = torch.ones(b, 2)
        center[..., 0] = (image_t.shape[3] - 1) / 2
        center[..., 1] = (image_t.shape[2] - 1) / 2
        M = kornia.get_rotation_matrix2d(center, angle, scale).to(device)
        rotated_image = kornia.warp_affine(image_t.float(), M, dsize=(h, w))
        return rotated_image

    return inner


def compose(transforms: Iterable[Callable]) -> Callable:
    """Helper function to compose transformations

    :param transforms: Iterable of transformation functions.
    :type transforms: Iterable[Callable]
    :return: Transform function
    :rtype: Callable
    """
    transforms = list(transforms)

    def inner(x):
        for transform in transforms:
            x = transform(x)
        return x

    return inner


def _roundup(value: Union[int, float, Union[List, Tuple, np.ndarray]]) -> Union[int, Union[List, Tuple, np.ndarray]]:
    """Computes ceiling of number or elementwise of a list, tuple, or np.ndarray

    :param value: number or Union[List, Tuple, np.ndarray] of numbers
    :type value: float, int or Union[List, Tuple, np.ndarray] of floats or ints
    :return: ceiling of input
    :rtype: int or Union[List, Tuple, np.ndarray] of int
    """
    return np.ceil(value).astype(int)


def _rads2angle(
    angle: float, 
    units: str,
) -> float:
    """Converts input angle from radians to degrees. If units is "degrees" then this is a no-op.

    :param angle: Input angle
    :type angle: float
    :param units: "radians" or "degrees", accepts synonyms "rads" and "rad" too.
    :type units: str
    :raises ValueError: Unrecognizes unit
    :return: Converted angle in degrees
    :rtype: float
    """
    if units.lower() == "degrees":
        return angle
    elif units.lower() in ["radians", "rads", "rad"]:
        angle = angle * 180.0 / np.pi
        return angle
    else:
        raise ValueError(f"Unrecognized units {units}")


def normalize() -> Callable:
    """Returns transform that performs the ImageNet normalization for torchvision models
    
    See https://pytorch.org/vision/stable/models.html

    :return: Transform function
    :rtype: Callable
    """
    normal = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def inner(image_t):
        return torch.stack([normal(t) for t in image_t])

    return inner


def preprocess_inceptionv1() -> Callable:
    """Returns preprocessing function to prepare input for original Tensorflow's InceptionV1 model.

    InceptionV1 takes in values from [-117, 138] so the preprocessing function takes in values from 0-1
    and maps them to [-117, 138]

    See https://github.com/tensorflow/lucid/blob/master/lucid/modelzoo/other_models/InceptionV1.py#L56 for details.
    Thanks to ProGamerGov for this!

    :return: Preprocessing function
    :rtype: Callable
    """
    return lambda x: x * 255 - 117


# define a set of standard transforms
standard_transforms = [
    pad(12, mode="constant", constant_value=0.5),
    jitter(8),
    random_scale([1 + (i - 5) / 50.0 for i in range(11)]),
    random_rotate(list(range(-10, 11)) + 5 * [0]),
    jitter(4),
]
