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

"""Util module of GraphVite"""
from __future__ import print_function, absolute_import

import os
import logging
from time import time
from functools import wraps

logger = logging.getLogger(__name__)

def recursive_default(obj, default):
    if isinstance(default, dict):
        new_obj = {}
        for key in default:
            if key in obj:
                new_obj[key] = recursive_default(obj[key], default[key])
            else:
                new_obj[key] = default[key]
        return type(default)(new_obj)
    else:
        return obj


def recursive_map(obj, function):
    if isinstance(obj, dict):
        return type(obj)({k: recursive_map(v, function) for k, v in obj.items()})
    elif isinstance(obj, list):
        return type(obj)([recursive_map(x, function) for x in obj])
    else:
        return function(obj)


class chdir(object):
    """
    Context manager for working directory.

    Parameters:
        dir (str): new working directory
    """
    def __init__(self, dir):
        self.dir = dir

    def __enter__(self):
        self.old_dir = os.getcwd()
        os.chdir(self.dir)

    def __exit__(self, *args):
        os.chdir(self.old_dir)


class Monitor(object):
    """
    Function call monitor.

    Parameters:
        name_style (str): style of displayed function name,
            can be `full`, `class` or `func`
    """

    def __init__(self, name_style="class"):
        assert name_style in ["full", "class", "func"]
        self.name_style = name_style

    def get_name(self, function, instance):
        is_method = len(function.__code__.co_varnames) > 0 and function.__code__.co_varnames[0] == "self"
        if self.name_style == "func" or not is_method:
            return "%s" % function.__name__
        if self.name_style == "class":
            return "%s.%s" % (instance.__class__.__name__, function.__name__)
        if self.name_style == "full":
            return "%s.%s.%s" % (instance.__module__, instance.__class__.__name__, function.__name__)

    def time(self, function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            name = self.get_name(function, args[0])
            start = time()
            result = function(*args, **kwargs)
            end = time()
            logger.info("[time] %s: %g s" % (name, end - start))
            return result

        return wrapper

    def call(self, function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            name = self.get_name(function, args[0])
            strings = ["%s" % repr(arg) for arg in args]
            strings += ["%s=%s" % (k, repr(v)) for k, v in kwargs.items()]
            logger.info("[call] %s(%s)" % (name, ", ".join(strings)))
            return function(*args, **kwargs)

        return wrapper

    def result(self, function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            name = self.get_name(function, args[0])
            strings = ["%s" % repr(arg) for arg in args]
            strings += ["%s=%s" % (k, repr(v)) for k, v in kwargs.items()]
            result = function(*args, **kwargs)
            logger.info("[result] %s(%s) = %s" % (name, ", ".join(strings), result))
            return result

        return wrapper

monitor = Monitor()