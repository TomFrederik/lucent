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

import warnings
from collections import OrderedDict
from typing import Callable, Iterable

import einops
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from lucent.misc.io import show
from lucent.optvis import objectives, transform, param


def render_vis(
    model: torch.nn.Module,
    objective_f: Callable,
    param_f: Optional[Callable] = None,
    optimizer: Optional[Callable[Any, torch.optim.Optimizer]] = None,
    transforms: Optional[Iterable[Callable[torch.Tensor, torch.Tensor]]] = None,
    thresholds: Iterable[int] = (512,),
    verbose: Optional[bool] = False,
    preprocess: Optional[bool] = True,
    preprocess_f: Optional[Callable[torch.Tensor, torch.Tensor]] = None,
    progress: Optional[bool] = True,
    show_image: Optional[bool] = True,
    save_image: Optional[bool] = False,
    image_name: Optional[str] = None,
    show_inline: Optional[bool] = False,
    fixed_image_size: Optional[bool] = None,
):
    """Main function for optimizing a given objective

    :param model: [description]
    :type model: torch.nn.Module
    :param objective_f: [description]
    :type objective_f: Callable
    :param param_f: [description], defaults to None
    :type param_f: Optional[Callable], optional
    :param optimizer: [description], defaults to None
    :type optimizer: Optional[Callable[Any, torch.optim.Optimizer]], optional
    :param transforms: [description], defaults to None. None is translated to transform.standard_transforms. To use no transforms at all supply an empty list.
    :type transforms: Optional[Iterable[Callable[torch.Tensor, torch.Tensor]]], optional
    :param thresholds: [description], defaults to (512,)
    :type thresholds: Iterable[int], optional
    :param verbose: [description], defaults to False
    :type verbose: Optional[bool], optional
    :param preprocess: [description], defaults to True
    :type preprocess: Optional[bool], optional
    :param progress: [description], defaults to True
    :type progress: Optional[bool], optional
    :param show_image: [description], defaults to True
    :type show_image: Optional[bool], optional
    :param save_image: [description], defaults to False
    :type save_image: Optional[bool], optional
    :param image_name: [description], defaults to None
    :type image_name: Optional[str], optional
    :param show_inline: [description], defaults to False
    :type show_inline: Optional[bool], optional
    :param fixed_image_size: [description], defaults to None
    :type fixed_image_size: Optional[bool], optional
    :return: [description]
    :rtype: [type]
    """
    if param_f is None:
        param_f = lambda: param.image(128)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()

    if optimizer is None:
        optimizer = lambda params: torch.optim.Adam(params, lr=5e-2)
    optimizer = optimizer(params)

    if transforms is None:
        transforms = transform.standard_transforms
    transforms = transforms.copy()

    if preprocess:
        if model._get_name() == "InceptionV1":
            # Original Tensorflow InceptionV1 takes input range [-117, 138]
            transforms.append(transform.preprocess_inceptionv1())
        else:
            if preprocess_f is None:
                # Assume we use normalization for torchvision.models
                # See https://pytorch.org/vision/stable/models.html
                transforms.append(transform.normalize())
            else:
                transforms.append(preprocess_f)

    #TODO make work for other sizes
    # Upsample images smaller than 224
    image_shape = image_f().shape
    if fixed_image_size is not None:
        new_size = fixed_image_size
    elif image_shape[2] < 224 or image_shape[3] < 224:
        new_size = 224
    else:
        new_size = None
    if new_size:
        transforms.append(
            torch.nn.Upsample(size=new_size, mode="bilinear", align_corners=True)
        )

    transform_f = transform.compose(transforms)

    hook = hook_model(model, image_f)
    objective_f = objectives.as_objective(objective_f)

    if verbose:
        model(transform_f(image_f()))
        print("Initial loss: {:.3f}".format(objective_f(hook)))

    images = []
    try:
        for i in tqdm(range(1, max(thresholds) + 1), disable=(not progress)):
            def closure():
                optimizer.zero_grad()
                try:
                    model(transform_f(image_f()))
                except RuntimeError as ex:
                    if i == 1:
                        # Only display the warning message
                        # on the first iteration, no need to do that
                        # every iteration
                        warnings.warn(
                            "Some layers could not be computed because the size of the "
                            "image is not big enough. It is fine, as long as the non"
                            "computed layers are not used in the objective function"
                            f"(exception details: '{ex}')"
                        )
                loss = objective_f(hook)
                loss.backward()
                return loss
                
            optimizer.step(closure)
            if i in thresholds:
                image = tensor_to_img_array(image_f())
                if verbose:
                    print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
                    if show_inline:
                        show(image)
                images.append(image)
    except KeyboardInterrupt:
        print("Interrupted optimization at step {:d}.".format(i))
        if verbose:
            print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
        images.append(tensor_to_img_array(image_f()))

        # re-raising error so that you can exit prematurely
        interrupt = None
        while interrupt not in ['y', 'n']:
            interrupt = input('Do you want to stop all queued-up optimizations as well? (y/n)')
        if interrupt == 'y':
            raise KeyboardInterrupt('Cancelling all queued-up optimizations...')

    if save_image:
        export(image_f(), image_name)
    if show_inline:
        show(tensor_to_img_array(image_f()))
    elif show_image:
        view(image_f())
    return images


def tensor_to_img_array(tensor: torch.Tensor) -> np.ndarray:
    """Converts tensor to image with channel-last ordering

    :param tensor: tensor that should be converted
    :type tensor: torch.Tensor
    :return: image with channel-last ordering
    :rtype: np.ndarray
    """
    image = tensor.cpu().detach().numpy()
    image = einops.rearrange(image, 'b c h w -> b h w c')
    return image


def view(tensor: torch.Tensor) -> None:
    """Displays tensor on screen by converting it to an image first

    :param tensor: tensor to be displayed
    :type tensor: torch.Tensor
    :raises ValueError: image has invalid shape
    """
    image = tensor_to_img_array(tensor)
    
    # check image shape
    if len(image.shape) not in (3, 4):
        raise ValueError(f"Image should have 3 or 4 dimensions, but has shape {image.shape}")

    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()


def export(tensor: torch.Tensor, image_name: Optional[str] = "image.jpg") -> None:
    """Saves tensor as image to disk under image_name.

    :param tensor: tensor to be saved
    :type tensor: torch.Tensor
    :param image_name: absolute path to image, defaults to "image.jpg"
    :type image_name: Optional[str], optional
    :raises ValueError: image has invalid shape
    """
    image = tensor_to_img_array(tensor)
    
    # check image shape
    if len(image.shape) not in (3, 4):
        raise ValueError(f"Image should have 3 or 4 dimensions, but has shape {image.shape}")

    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).save(image_name)


