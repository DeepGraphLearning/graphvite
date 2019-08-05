# Copyright 2019 MilaGraph. All Rights Reserved.
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
#
# Author: Zhaocheng Zhu

"""Optimizer module of GraphVite"""
from __future__ import absolute_import

import sys

from . import lib, auto
from .helper import find_all_names

module = sys.modules[__name__]

class Optimizer(object):
    """
    Optimizer(type=auto, *args, **kwargs)
    Create an optimizer instance of any type.

    Parameters:
        type (str or auto): optimizer type,
            can be 'SGD', 'Momentum', 'AdaGrad', 'RMSprop' or 'Adam'
    """
    def __new__(cls, type=auto, *args, **kwargs):
        if type == auto:
            return lib.optimizer.Optimizer(auto)
        elif hasattr(lib.optimizer, type):
            return getattr(lib.optimizer, type)(*args, **kwargs)
        else:
            raise ValueError("Unknown optimizer `%s`" % type)


for name in find_all_names(lib.optimizer):
    if name not in module.__dict__:
         Class = getattr(lib.optimizer, name)
         # transfer module ownership so that autodoc can work
         Class.__module__ = Class.__module__.replace("libgraphvite", "graphvite")
         module.__dict__[name] = Class

__all__ = [
    "Optimizer",
    "LRSchedule",
    "SGD", "Momentum", "AdaGrad", "RMSprop", "Adam"
]