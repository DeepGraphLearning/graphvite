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

"""GraphVite: graph embedding at high speed and large scale"""
from __future__ import absolute_import, unicode_literals

import os
import sys
import imp
import logging

from . import util

package_path = os.path.dirname(__file__)
candidate_paths = [
    os.path.realpath(os.path.join(package_path, "lib")),
    os.path.realpath(os.path.join(package_path, "../../lib")),
    os.path.realpath(os.path.join(package_path, "../../build/lib"))
]
lib_file = imp.find_module("libgraphvite", candidate_paths)[1]
lib_path = os.path.dirname(lib_file)
with util.chdir(lib_path):
    lib = imp.load_dynamic("libgraphvite", lib_file)

from libgraphvite import dtype, auto, __version__

from . import base
from .base import init_logging
cfg = base.load_global_config()
base.init_logging(logging.INFO)

from . import helper
from . import graph, solver, optimizer
from . import dataset

module = sys.modules[__name__]
module.__dict__.update(dtype.__members__)