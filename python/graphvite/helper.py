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

"""Helper functions for loading C++ extension"""
from __future__ import absolute_import, print_function

import re

from . import lib
lib.name2dtype = {n: t for t, n in lib.dtype2name.items()}


def signature(name, *args):
    strings = [name]
    for arg in args:
        if isinstance(arg, lib.dtype):
            strings.append(lib.dtype2name[arg])
        else:
            strings.append(str(arg))
    return "_".join(strings)


def find_all_names(module):
    pattern = re.compile("[^_]+")
    names = []
    for name in module.__dict__:
        if pattern.match(name):
            names.append(name)
    return names


def find_all_templates(module):
    pattern = re.compile("([^_]+)(?:_[^_]+)+")
    names = set()
    for full_name in module.__dict__:
        result = pattern.match(full_name)
        if result:
            names.add(result.group(1))
    return list(names)


def get_any_instantiation(module, name):
    pattern = re.compile("%s(?:_[^_]+)+" % name)
    for full_name in module.__dict__:
        if pattern.match(full_name):
            return getattr(module, full_name)


def get_instantiation_info(module, name, template_keys):
    pattern = re.compile("%s((?:_[^_]+)+)" % name)
    possible_parameters = []
    for full_name in module.__dict__:
        result = pattern.match(full_name)
        if result:
            possible_parameters.append(result.group(1).split("_")[1:])
    template_values = zip(*possible_parameters)

    infos = ["Instantiations:"]
    for key, values in zip(template_keys, template_values):
        values = list(set(values))
        if values[0] in lib.name2dtype:
            values = [lib.name2dtype[v] for v in values]
        else:
            values = sorted(eval(v) for v in values)
        values = [str(v) for v in values]
        infos.append("- **%s**: %s" % (key, ", ".join(values)))
    return "\n    ".join(infos)


class TemplateHelper(object):

    def __new__(cls, *args, **kwargs):
        args = list(args)
        parameters = []
        for i, key in enumerate(cls.template_keys):
            if args:
                parameters.append(args.pop(0))
            elif key in kwargs:
                parameters.append(kwargs.pop(key))
            else:
                value = cls.template_values[i]
                if value is None:
                    raise TypeError("Required argument `%s` (pos %d) not found" % (key, i))
                else:
                    parameters.append(value)

        full_name = signature(cls.name, *parameters)
        if hasattr(cls.module, full_name):
            return getattr(cls.module, full_name)(*args, **kwargs)
        else:
            strings = ["%s=%s" % (k, v) for k, v in zip(cls.template_keys, parameters)]
            raise AttributeError("Can't find an instantiation of %s with %s" % (cls.name, ", ".join(strings)))


def make_helper_class(module, name, target_module, template_keys, template_values):
    InstanceClass = get_any_instantiation(module, name)
    # copy all members so that autodoc can work
    members = dict(InstanceClass.__dict__)
    # add instantiation info to docstring
    doc = InstanceClass.__doc__
    indent = re.search("\n *", doc).group(0)
    info = "\n" + get_instantiation_info(module, name, template_keys)
    doc += info.replace("\n", indent)
    members.update({
        "module": module,
        "name": name,
        "__module__": target_module.__name__,
        "__doc__": doc,
        "template_keys": template_keys,
        "template_values": template_values
    })
    TemplateClass = type(name, (TemplateHelper,), members)
    return TemplateClass