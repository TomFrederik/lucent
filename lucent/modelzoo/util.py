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

"""Utility functions for modelzoo models."""

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from typing import List, Optional, Union


def get_model_layers(
    model: torch.nn.Module, 
    getLayerRepr: Optional[bool] = False
) -> Union[List[str], OrderedDict[str, str]]:

    """Get the names of all layers of a network. The names are given in the format that can be used
       to access them via objectives.

    :param model: the network to get the names of
    :param getLayerRepr: whether to return a OrderedDict of layer names, layer representation string pair. If False just return a list of names.
    :type model: torch.nn.Module
    :raises ValueError: model has wrong type
    :raises ValueError: model has no modules
    :return: dict of name, repr pairs or just list of names of all layers (including activations if they are instantiated as layers)
    :rtype: Union[List[str], OrderedDict[str, str]]
    """
    
    # check input
    if not isinstance(model, torch.nn.Module):
        raise ValueError(f"model should have type torch.nn.Module but has type {type(model)}")

    layers = OrderedDict() if getLayerRepr else []
    # recursive function to get names
    def recursive_get_names(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                
                # only add actual layers and not group name itself
                if (
                    not isinstance(layer, nn.Sequential)
                    and not isinstance(layer, nn.ModuleDict)
                    and not isinstance(layer, nn.ModuleList)
                ):
                    if getLayerRepr:
                        layers["_".join(prefix+[name])] = layer.__repr__()
                    else:
                        layers.append("_".join(prefix + [name]))
                
                # recurse
                recursive_get_names(layer, prefix=prefix + [name])
        else:   
            raise ValueError('net has no _modules attribute! Check if your model is properly instantiated..')
    
    recursive_get_names(model)
    
    return layers