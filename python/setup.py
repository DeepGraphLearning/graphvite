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

from __future__ import print_function, absolute_import

import os
from setuptools import setup, find_packages

from graphvite import __version__, lib_path, lib_file

name = "graphvite"
faiss_file = os.path.join(lib_path, "libfaiss.so")
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# library files
install_path = os.path.join(name, "lib")
data_files = [(install_path, [lib_file, faiss_file])]
# configuration files
for path, dirs, files in os.walk(os.path.join(project_path, "config")):
    install_path = os.path.join(name, os.path.relpath(path, project_path))
    files = [os.path.join(path, file) for file in files]
    data_files.append((install_path, files))

setup(
    name=name,
    version=__version__,
    description="A general and high-performance graph embedding system for various applications",
    packages=find_packages(),
    data_files=data_files,
    entry_points={"console_scripts": ["graphvite = graphvite.cmd:main"]},
    zip_safe=False,
    #install_requires=["numpy", "pyyaml", "easydict", "six", "future"],
    #extras_requires={"app": ["imageio", "psutil", "scipy", "matplotlib", "torch", "torchvision", "nltk"]}
)