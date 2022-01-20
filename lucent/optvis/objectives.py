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

from __future__ import absolute_import, division, print_function, annotations

from typing import Iterable, TypeVar, Callable, Optional, Any, Union

from decorator import decorator
import numpy as np
import torch
import torch.nn.functional as F

from lucent.optvis.objectives_util import _make_arg_str, _extract_act_pos, _T_handle_batch

T = TypeVar('T')

#TODO: restructure this
# premade objectives should go in their own submodule
# utils in a different one
# baseclass + decorator in its own?


class Objective:
    """Base class for objectives. Implements basic arithmetic on objectives.
    """
    def __init__(
        self, 
        objective_func: Callable[[torch.nn.Module], Any], 
        name: Optional[str] = "", 
        description: Optional[str] = "",
    ) -> None:
    #TODO this docstring is not displayed in library, maybe if I remove the docstring of Objectives itself?
        """Constructor for Objective

        :param objective_func: Objective function that evaluates an objective on a given model
        :type objective_func: Callable[[torch.nn.Module], Any]
        :param name: Name of the objective, defaults to ""
        :type name: Optional[str], optional
        :param description: Description of the objective, defaults to ""
        :type description: Optional[str], optional
        """
        self.objective_func = objective_func
        self.name = name
        self.description = description

    def __call__(self, model):
        return self.objective_func(model)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            objective_func = lambda model: other + self(model)
            name = self.name
            description = self.description
        else:
            objective_func = lambda model: self(model) + other(model)
            name = ", ".join([self.name, other.name])
            description = "Sum(" + " +\n".join([self.description, other.description]) + ")"
        return Objective(objective_func, name=name, description=description)

    @staticmethod
    def sum(objs: Iterable[Objective]) -> Objective:
        #TODO clean up this docstring explanation a bit
        """Alternative to sum(objs) which would return a nested description Sum(d1 + Sum(d2 + Sum(...))) for descriptions d_i which is unreadable.
        Using this method will produce description Sum(d1 + d2 + ...) instead.
        To call this, do Objective.sum(objs).

        :param objs: The objectives that should be summed.
        :type objs: Iterable[Objective]
        :return: New Objective instance with the sum of the objective functions as objective function.
        :rtype: Objective
        """
        # convert to list, otherwise iterators might be empty after first pass
        objs = list(objs)

        objective_func = lambda T: sum([obj(T) for obj in objs])
        descriptions = [obj.description for obj in objs]
        description = "Sum(" + " +\n".join(descriptions) + ")"
        names = [obj.name for obj in objs]
        name = ", ".join(names)
        return Objective(objective_func, name=name, description=description)

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-1 * other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            objective_func = lambda model: other * self(model)
            return Objective(objective_func, name=self.name, description=self.description)
        else:
            # Note: In original Lucid library, objectives can be multiplied with non-numbers
            # Removing for now until we find a good use case
            raise TypeError('Can only multiply by int or float. Received type ' + str(type(other)))

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(1 / other)
        else:
            raise TypeError('Can only divide by int or float. Received type ' + str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    # need this for summation to work properly
    def __radd__(self, other):
        return self.__add__(other)


def wrap_objective() -> Callable[[Callable], Objective]:
    """Decorator to construct objectives from function definitions

    :return: Objective that is constructed from the decorated function
    :rtype: Callable[[Callable], Objective]
    """
    @decorator
    def inner(func, *args, **kwds):
        objective_func = func(*args, **kwds)
        objective_name = func.__name__
        args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
        description = objective_name.title() + args_str
        return Objective(objective_func, objective_name, description)
    return inner

#TODO figure out what exactly this thing here does
def handle_batch(batch=None):
    return lambda f: lambda model: f(_T_handle_batch(model, batch=batch))


@wrap_objective()
def neuron(
    layer: str, 
    n_channel: int, 
    x: Optional[int] = None, 
    y: Optional[int] = None, 
    batch: Optional[int] = None,
) -> Objective: #TODO Fix the ASCII art in the docstring: It's not displayed properly on the RTD website
    """Visualize a single neuron of a single channel.

    Defaults to the center neuron. When width and height are even numbers, we
    choose the neuron in the bottom right of the center 2x2 neurons.

    Odd width & height:               Even width & height:

    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   | X |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   | X |   |
    +---+---+---+                     +---+---+---+---+
                                      |   |   |   |   |
                                      +---+---+---+---+

    :param layer: Name of the layer
    :type layer: str
    :param n_channel: Channel/neuron number
    :type n_channel: int
    :param x: x-position, defaults to None
    :type x: Optional[int], optional
    :param y: y-position, defaults to None
    :type y: Optional[int], optional
    :param batch: which position at the batch dimension of the image tensor this objective is applied to, defaults to None
    :type batch: Optional[int], optional
    :return: Objective to optimize input for a single position of a single channel
    :rtype: Objective
    """
    
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y)
        return -layer_t[:, n_channel].mean()
    return inner


@wrap_objective()
def channel(
    layer: str, 
    n_channel: int, 
    batch: Optional[int] = None,
) -> Objective:
    """Visualize a single channel

    :param layer: Name of the layer
    :type layer: str
    :param n_channel: Channel number
    :type n_channel: int
    :param batch: which position at the batch dim of the image tensor this objective is applied to, defaults to None
    :type batch: Optional[int], optional
    :return: Objective to optimize input for a single channel
    :rtype: Objective
    """
    @handle_batch(batch)
    def inner(model):
        return -model(layer)[:, n_channel].mean()
    return inner

@wrap_objective()
def neuron_weight(
    layer: str, 
    weight: torch.Tensor, 
    x: Optional[int] = None, 
    y: Optional[int] = None, 
    batch: Optional[int] = None,
) -> Objective:
    """Linearly weighted channel activation at one location as objective

    :param layer: Name of the layer
    :type layer: str
    :param weight: A torch.Tensor of same length as the number of channels
    :type weight: torch.Tensor
    :param x: x-position, defaults to None
    :type x: Optional[int], optional
    :param y: y-position, defaults to None
    :type y: Optional[int], optional
    :param batch: which position at the batch dimension of the image tensor this objective is applied to, defaults to None
    :type batch: Optional[int], optional
    :return: Objective to optimize input for a linearly weighted channel activation at one location
    :rtype: Objective
    """
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y)
        if weight is None:
            return -layer_t.mean()
        else:
            return -(layer_t.squeeze() * weight).mean()
    return inner

@wrap_objective()
def channel_weight(
    layer: str, 
    weight: torch.Tensor, 
    batch: Optional[int] = None,
) -> Objective:
    """ Linearly weighted channel activation as objective
    
    :param layer: Name of the layer
    :type layer: str
    :param weight: A torch.Tensor of same length as the number of channels
    :type weight: torch.Tensor
    :param batch: which position at the batch dim of the image tensor this objective is applied to, defaults to None
    :type batch: Optional[int], optional
    :return: Objective to optimize input for a linearly weighted channel activation
    :rtype: Objective
    """
    #TODO add option to normalize the weight vector and set default to True?

    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        return -(layer_t * weight.view(1, -1, 1, 1)).mean()
    return inner

@wrap_objective()
def localgroup_weight(
    layer: str, 
    weight: Optional[torch.Tensor] = None, 
    x: Optional[int] = None, 
    y: Optional[int]=None, 
    wx: Optional[int] = 1, 
    wy: Optional[int] = 1, 
    batch: Optional[int] = None,
) -> Objective:
    """Linearly weighted channel activation around some spot as objective

    :param layer: Name of the layer
    :type layer: str
    :param weight: A torch.Tensor of same length as the number of channels, defaults to None
    :type weight: Optional[torch.Tensor], optional
    :param x: x-position, defaults to None
    :type x: Optional[int], optional
    :param y: y-position, defaults to None
    :type y: Optional[int], optional
    :param wx: window size in x-direction, defaults to 1
    :type wx: Optional[int], optional
    :param wy: window size in y-direction, defaults to 1
    :type wy: Optional[int], optional
    :param batch: which position at the batch dimension of the image tensor this objective is applied to, defaults to None
    :type batch: Optional[int], optional
    :return: Objective to optimize linearly weighted channel activation around some spot
    :rtype: Objective
    """
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        if weight is None:
            return -(layer_t[:, :, y:y + wy, x:x + wx]).mean()
        else:
            return -(layer_t[:, :, y:y + wy, x:x + wx] * weight.view(1, -1, 1, 1)).mean()
    return inner

#TODO this and channel weight should be merged together, I think they do basically the same thing?
#TODO would need to add an option for the loss function though (cosine vs wheighted)
@wrap_objective()
def direction(
    layer: str, 
    direction: torch.Tensor, 
    batch: Optional[int] = None,
) -> Objective:
    """Visualize a direction in activation space.

    :param layer: Name of the layer
    :type layer: str
    :param direction: torch.Tensor of shape (num_channels, ) giving the direction
    :type direction: torch.Tensor
    :param batch: which position at the batch dimension of the image tensor this objective is applied to, defaults to None
    :type batch: Optional[int], optional
    :return: Objective to optimize input for a particular direction in activation space.
    :rtype: Objective
    """
    #TODO add option to normalize the direction vector and set default to True?
    @handle_batch(batch)
    def inner(model):
        return -torch.nn.CosineSimilarity(dim=1)(direction.reshape(
            (1, -1, 1, 1)), model(layer)).mean()

    return inner


@wrap_objective()
def direction_neuron(
    layer: str,
    direction: torch.Tensor,
    x: Optional[int] = None,
    y: Optional[int] = None,
    batch: Optional[int] = None,
) -> Objective:
    """Visualize a single (x, y) position along the given direction

    Similar to the neuron objective, defaults to the center neuron.

    :param layer: Name of layer
    :type layer: str
    :param direction: torch.Tensor of shape (num_channels, ) that gives the direction optimize
    :type direction: torch.Tensor
    :param x: x-position, defaults to None
    :type x: Optional[int], optional
    :param y: y-position, defaults to None
    :type y: Optional[int], optional
    :param batch: which position at the batch dimension of the image tensor this objective is applied to, defaults to None
    :type batch: Optional[int], optional
    :return: Objective to optimize input for a particular direction in activation space at a single position
    :rtype: Objective
    """

    @handle_batch(batch)
    def inner(model):
        # breakpoint()
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y)
        return -torch.nn.CosineSimilarity(dim=1)(direction.reshape(
            (1, -1, 1, 1)), layer_t).mean()

    return inner


#TODO What does this do and shouldn't it be in some other file?
def _torch_blur(tensor, out_c=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    depth = tensor.shape[1]
    weight = np.zeros([depth, depth, out_c, out_c])
    for ch in range(depth):
        weight_ch = weight[ch, ch, :, :]
        weight_ch[ :  ,  :  ] = 0.5
        weight_ch[1:-1, 1:-1] = 1.0
    weight_t = torch.tensor(weight).float().to(device)
    conv_f = lambda t: F.conv2d(t, weight_t, None, 1, 1)
    return conv_f(tensor) / conv_f(torch.ones_like(tensor))


@wrap_objective()
def blur_input_each_step() -> Objective:
    """Minimizing this objective is equivelant to blurring input each step.
    Optimizing (-k)*blur_input_each_step() is equivalant to:
    input <- (1-k)*input + k*blur(input)
    An operation that was used in early feature visualization work.
    See Nguyen, et al., 2015.

    :return: Objective which optimizes input to be blurred.
    :rtype: Objective
    """
    def inner(model):
        model_input = model("input")
        with torch.no_grad():
            model_input_blurred = _torch_blur(model_input)
        return -0.5*torch.sum((model_input - model_input_blurred)**2)
    return inner


@wrap_objective()
def channel_interpolate(
    layer1: str, 
    n_channel1: int, 
    layer2: str, 
    n_channel2: int,
) -> Objective:
    """Interpolate between layer1, n_channel1 and layer2, n_channel2.
    Optimize for a convex combination of layer1, n_channel1 and
    layer2, n_channel2, transitioning across the batch.

    :param layer1: layer to optimize 100% at batch=0.
    :type layer1: str
    :param n_channel1: neuron index to optimize 100% at batch=0.
    :type n_channel1: int
    :param layer2: layer to optimize 100% at batch=N.
    :type layer2: str
    :param n_channel2: neuron index to optimize 100% at batch=N.
    :type n_channel2: int
    :return: Objective to optimizes input towards the channel interpolation between the given channels
    :rtype: Objective
    """
    
    def inner(model):
        batch_n = list(model(layer1).shape)[0]
        arr1 = model(layer1)[:, n_channel1]
        arr2 = model(layer2)[:, n_channel2]
        weights = np.arange(batch_n) / (batch_n - 1)
        sum_loss = 0
        for n in range(batch_n):
            sum_loss -= (1 - weights[n]) * arr1[n].mean()
            sum_loss -= weights[n] * arr2[n].mean()
        return sum_loss
    return inner


@wrap_objective()
def alignment(
    layer: str, 
    decay_ratio: Optional[float] = 2,
) -> Objective:
    """Encourage neighboring images to be similar.
    When visualizing the interpolation between two objectives, it's often
    desirable to encourage analogous objects to be drawn in the same position,
    to make them more comparable.
    This term penalizes L2 distance between neighboring images, as evaluated at
    layer.
    In general, we find this most effective if used with a parameterization that
    shares across the batch. (In fact, that works quite well by itself, so this
    function may just be obsolete.)

    :param layer: layer to penalize at.
    :type layer: str
    :param decay_ratio: how much to decay penalty as images move apart in batch., defaults to 2
    :type decay_ratio: Optional[float], optional
    :return: Objective to optimize input towards alignment across batch dimension
    :rtype: Objective
    """
    def inner(model):
        batch_n = list(model(layer).shape)[0]
        layer_t = model(layer)
        accum = 0
        for d in [1, 2, 3, 4]:
            for i in range(batch_n - d):
                a, b = i, i + d
                arr_a, arr_b = layer_t[a], layer_t[b]
                accum += ((arr_a - arr_b) ** 2).mean() / decay_ratio ** float(d)
        return accum
    return inner

@wrap_objective()
def diversity(layer: str) -> Objective:
    """Encourage diversity between each batch element.

    A neural net feature often responds to multiple things, but naive feature
    visualization often only shows us one. If you optimize a batch of images,
    this objective will encourage them all to be different.

    In particular, it calculates the correlation matrix of activations at layer
    for each image, and then penalizes cosine similarity between them. This is
    very similar to ideas in style transfer, except we're *penalizing* style
    similarity instead of encouraging it.

    :param layer: layer to evaluate activation correlations on.
    :type layer: str
    :return: Objective that encourages input towards diversity
    :rtype: Objective
    """
    def inner(model):
        layer_t = model(layer)
        batch, channels, _, _ = layer_t.shape
        flattened = layer_t.view(batch, channels, -1)
        grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
        grams = F.normalize(grams, p=2, dim=(1, 2))
        return -sum([ sum([ (grams[i]*grams[j]).sum()
               for j in range(batch) if j != i])
               for i in range(batch)]) / batch
    return inner



#TODO is it good that this also leaves callables unchanged? Seems to me like
#     it would be better if it converts everything to instances of Objective
def as_objective(obj: Union[Objective, Callable, str]) -> Union[Callable, Objective]:
    """Strings of the form "layer:n" become the Objective channel(layer, n).
    Objectives and Callables are returned unchanged.

    :param obj: Objective, Callable or layer:channel string
    :type obj: Union[Objective, Callable, str]
    :return: Objective instance or Callable representing the objective function
    :rtype: Union[Callable, Objective]
    """
    if isinstance(obj, Objective):
        return obj
    if callable(obj):
        return obj
    if isinstance(obj, str):
        layer, chn = obj.split(":")
        layer, chn = layer.strip(), int(chn)
        return channel(layer, chn)
