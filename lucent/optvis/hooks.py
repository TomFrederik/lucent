from collections import OrderedDict
from typing import Callable

import torch

class ModuleHook:
    """Hook class to save features computed during forward pass of given module
    """
    def __init__(self, module: torch.nn.Module) -> None:
        """Constructor for ModuleHook

        :param module: Module to which the hook should be attached. In practice, this is usually a layer within a larger Module.
        :type module: torch.nn.Module
        :return: None
        """
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(
        self, 
        module: torch.nn.Module, 
        input: torch.Tensor, 
        output: torch.Tensor
    ) -> None:
        """Hooking function that stores the hooked module's output (its features)

        :param module: hooked module
        :type module: torch.nn.Module
        :param input: inpute to module
        :type input: torch.Tensor
        :param output: output of module on given input
        :type output: torch.Tensor
        """
        self.module = module
        self.features = output

    def close(self):
        self.hook.remove()


def hook_model(model: torch.nn.Module, image_f: Callable) -> Callable:
    """Creates hooks for every layer in a model for a given image function.

    :param model: the model to be hooked
    :type model: torch.nn.Module
    :param image_f: the function returning the parameters of the image and the image itself as a tensor
    :type image_f: Callable
    :raises TypeError: model is wrong type
    :raises TypeError: image_f is wrong type
    :return: ``hook`` that returns the models features at a given layer for the given image
    :rtype: Callable
    """
    
    # check inputs
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model should be type torch.nn.Module but is {type(model)}")
    if not isinstance(image_f, Callable):
        raise TypeError(f'image_f should be type Callable but is {type(image_f)}')

    features = OrderedDict()

    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                features["->".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])
        else:
            raise ValueError('net has not _modules attribute! Check if your model is properly instantiated..')

    hook_layers(model)

    def hook(layer: str) -> torch.Tensor:
        """Method to access stored output during forward pass

        :param layer: Identifier of the layer we want to access
        :type layer: str
        :return: Features (output) of the layer during its last forward pass.
        :rtype: torch.Tensor
        """
        if layer == "input":
            out = image_f()
        elif layer == "labels":
            out = list(features.values())[-1].features
        else:
            assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.model_utils.get_model_layers(model)`."
            out = features[layer].features
        assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        return out

    return hook