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

from collections import OrderedDict
from typing import Optional, Tuple, Callable, Iterator

import numpy as np
import torch


# TODO have no idea why this is being used.
class CompositeActivation(torch.nn.Module):
    def forward(self, x):
        x = torch.atan(x)
        return torch.cat([x/0.67, (x*x)/0.6], 1)


def cppn(
    size: int, 
    num_output_channels: Optional[int] = 3, 
    num_hidden_channels: Optional[int] = 24, 
    num_layers: Optionall[int] = 8,
    activation_fn: Optional[torch.nn.Module] = CompositeActivation, 
    normalize: Optional[bool] = False
) -> Tuple[Iterator, Callable]:
    """Creates cppn parameterization.

    :param size: image size
    :type size: int
    :param num_output_channels: number of output channels, defaults to 3
    :type num_output_channels: Optional[int], optional
    :param num_hidden_channels: number of hidden channels, defaults to 24
    :type num_hidden_channels: Optional[int], optional
    :param num_layers: number of layers, defaults to 8
    :type num_layers: Optionall[int], optional
    :param activation_fn: activation function after hidden layers, defaults to CompositeActivation
    :type activation_fn: Optional[torch.nn.Module], optional
    :param normalize: Whether to use instance normalization after each hidden layer, defaults to False
    :type normalize: Optional[bool], optional
    :return: Iterator over network parameters and a function that returns the network's output on the grid [-sqrt(3), sqrt(3)] with step size ``size``.
    :rtype: Tuple[Iterator, Callable]
    """

    r = 3 ** 0.5

    coord_range = torch.linspace(-r, r, size)
    x = coord_range.view(-1, 1).repeat(1, coord_range.size(0))
    y = coord_range.view(1, -1).repeat(coord_range.size(0), 1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.stack([x, y], dim=0).unsqueeze(0).to(device)

    layers = []
    kernel_size = 1
    for i in range(num_layers):
        out_c = num_hidden_channels
        in_c = out_c * 2 # * 2 for composite activation
        if i == 0:
            in_c = 2
        if i == num_layers - 1:
            out_c = num_output_channels
        layers.append(('conv{}'.format(i), torch.nn.Conv2d(in_c, out_c, kernel_size)))
        if normalize:
            layers.append(('norm{}'.format(i), torch.nn.InstanceNorm2d(out_c)))
        if i < num_layers - 1:
            layers.append(('actv{}'.format(i), activation_fn()))
        else:
            layers.append(('output', torch.nn.Sigmoid()))

    # Initialize model
    net = torch.nn.Sequential(OrderedDict(layers)).to(device)
    # Initialize weights
    def weights_init(module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, 0, np.sqrt(1/module.in_channels))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    net.apply(weights_init)
    # Set last conv2d layer's weights to 0
    torch.nn.init.zeros_(dict(net.named_children())['conv{}'.format(num_layers - 1)].weight)
    return net.parameters(), lambda: net(input_tensor)
