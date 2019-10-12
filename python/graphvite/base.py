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

from __future__ import absolute_import

import os
import sys
import yaml
import logging
from easydict import EasyDict

from . import lib, dtype
from .util import recursive_default, assert_in


root = os.path.expanduser("~/.graphvite")
if not os.path.exists(root):
    os.mkdir(root)

# default config
default = EasyDict()
default.backend = "graphvite"
default.dataset_path = os.path.join(root, "dataset")
default.float_type = dtype.float32
default.index_type = dtype.uint32


def load_global_config():
    config_file = os.path.join(root, "config.yaml")
    if os.path.exists(config_file):
        with open(config_file, "r") as fin:
            cfg = EasyDict(yaml.safe_load(fin))
        cfg = recursive_default(cfg, default)
    else:
        cfg = default

    assert_in(["graphvite", "torch"], backend=cfg.backend)
    if not os.path.exists(cfg.dataset_path):
        os.mkdir(cfg.dataset_path)
    if isinstance(cfg.float_type, str):
        cfg.float_type = eval(cfg.float_type)
    if isinstance(cfg.index_type, str):
        cfg.index_type = eval(cfg.index_type)

    return cfg


def init_logging(level=logging.INFO, dir="", verbose=False):
    """
    Init logging.

    Parameters:
        level (int, optional): logging level, INFO, WARNING, ERROR or FATAL
        dir (str, optional): log directory, leave empty for standard I/O
        verbose (bool, optional): verbose mode
    """
    logger = logging.getLogger(__package__)
    logger.level = level
    if dir == "":
        logger.handlers = [logging.StreamHandler(sys.stdout)]
    else:
        logger.handlers = [logging.FileHandler(os.path.join(dir, "log.txt"))]

    if level <= logging.INFO:
        lib.init_logging(lib.INFO, dir, verbose)
    elif level <= logging.WARNING:
        lib.init_logging(lib.WARNING, dir, verbose)
    elif level <= logging.ERROR:
        lib.init_logging(lib.ERROR, dir, verbose)
    else:
        lib.init_logging(lib.FATAL, dir, verbose)