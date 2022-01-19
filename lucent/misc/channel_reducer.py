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

"""Helper for using sklearn.decomposition on high-dimensional tensors.

Provides ChannelReducer, a wrapper around sklearn.decomposition to help them
apply to arbitrary rank tensors. It saves lots of annoying reshaping.
"""

from typing import Optional, Union, Callable, TypeVar

import numpy as np
import sklearn.decomposition
import torch

try:
    from sklearn.decomposition.base import BaseEstimator
except (AttributeError, ModuleNotFoundError):
    from sklearn.base import BaseEstimator

ArrayType = TypeVar('ArrayType', np.ndarray, torch.Tensor)

class ChannelReducer(object):
    def __init__(
        self, 
        n_components: Optional[int] = 3, 
        reduction_alg: Optional[Union[str, BaseEstimator]] = "NMF", 
        **kwargs,
    ):
        """Helper for dimensionality reduction to the innermost dimension of a tensor.

        This class wraps sklearn.decomposition classes to help them apply to arbitrary
        rank tensors. It saves lots of annoying reshaping.

        See the original sklearn.decomposition documentation:
        http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition

        :param n_components: Number of dimensions to reduce inner most dimension to, defaults to 3
        :type n_components: Optional[int], optional
        :param reduction_alg: A string or sklearn.decomposition class - one of "NMF", "PCA", "FastICA" and "MiniBatchDictionaryLearning", defaults to "NMF"
        :type reduction_alg: Optional[str], optional
        :raises ValueError: n_components is not an integer
        :raises ValueError: n_components <= 0
        :raises ValueError: Unknown reduction algorithm
        """
        if not isinstance(n_components, int):
            raise ValueError("n_components must be an int, not '%s'." % n_components)
        if n_components <= 0:
            raise ValueError("n_components must be strictly > 0")
        # Defensively look up reduction_alg if it is a string and give useful errors.
        algorithm_map = {}
        for name in dir(sklearn.decomposition):
            obj = sklearn.decomposition.__getattribute__(name)
            if isinstance(obj, type) and issubclass(obj, BaseEstimator):
                algorithm_map[name] = obj
        if isinstance(reduction_alg, str):
            if reduction_alg in algorithm_map:
                reduction_alg = algorithm_map[reduction_alg]
            else:
                raise ValueError(
                    "Unknown dimensionality reduction method '%s'." % reduction_alg
                )

        self.n_components = n_components
        self._reducer = reduction_alg(n_components=n_components, **kwargs)
        self._is_fit = False

    @classmethod
    def _apply_flat(
        cls, 
        f: Callable, 
        acts: ArrayType,
    ) -> ArrayType:
        """Utility for applying f to inner dimension of acts.
        Flattens acts into a 2D tensor, applies f, then unflattens so that all
        dimensions except innermost are unchanged.

        :param f: Reducer function to be applied
        :type f: Callable
        :param acts: Tensor that should be modified by f
        :type acts: ArrayType
        :return: Output of f with input acts
        :rtype: ArrayType
        """
        orig_shape = acts.shape
        acts_flat = acts.reshape([-1, orig_shape[-1]])
        new_flat = f(acts_flat)
        
        #TODO Why do we return [A B C] for ndarrays but [A*B C] for tensors? This seems very off.
        # I will comment this out and pray for the best
        # if not isinstance(new_flat, np.ndarray):
        #     return new_flat
        shape = list(orig_shape[:-1]) + [-1] 
        return new_flat.reshape(shape)

    def fit(self, acts):
        self._is_fit = True
        return ChannelReducer._apply_flat(self._reducer.fit, acts)

    def fit_transform(self, acts):
        self._is_fit = True
        return ChannelReducer._apply_flat(self._reducer.fit_transform, acts)

    def transform(self, acts):
        return ChannelReducer._apply_flat(self._reducer.transform, acts)

    def __call__(self, acts):
        if self._is_fit:
            return self.transform(acts)
        else:
            return self.fit_transform(acts)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name + "_" in self._reducer.__dict__:
            return self._reducer.__dict__[name + "_"]

    def __dir__(self):
        dynamic_attrs = [
            name[:-1]
            for name in dir(self._reducer)
            if name[-1] == "_" and name[0] != "_"
        ]

        return (
            list(ChannelReducer.__dict__.keys())
            + list(self.__dict__.keys())
            + dynamic_attrs
        )
