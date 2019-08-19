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

""""Command line executables of GraphVite"""
from __future__ import print_function, absolute_import

import os
import re
import yaml
import logging
import argparse
from easydict import EasyDict

import numpy as np

import graphvite as gv
import graphvite.application as gap


def get_config_path():
    candidate_paths = [
        os.path.join(gv.package_path, "config"),
        os.path.join(gv.package_path, "../../config")
    ]
    for config_path in candidate_paths:
        if os.path.isdir(config_path):
            return config_path
    raise IOError("Can't find baseline configuration directory. Did you install GraphVite correctly?")


def get_parser():
    parser = argparse.ArgumentParser(description="GraphVite command line executor v%s" % gv.__version__)
    command = parser.add_subparsers(metavar="command", dest="command")
    command.required = True

    run = command.add_parser("run", help="run from configuration file")
    run.add_argument("config", help="yaml configuration file")
    run.add_argument("--no-eval", help="turn off evaluation", dest="eval", action="store_false")
    run.add_argument("--gpu", help="override the number of GPUs", type=int)
    run.add_argument("--cpu", help="override the number of CPUs per GPU", type=int)

    visualize = command.add_parser("visualize", help="visualize high-dimensional vectors")
    visualize.add_argument("file", help="data file (numpy dump or txt)")
    visualize.add_argument("--label", help="label file (numpy dump or txt)")
    visualize.add_argument("--save", help="png or pdf file to save")
    visualize.add_argument("--perplexity", help="perplexity for the neighborhood", type=float, default=30)
    visualize.add_argument("--3d", help="3d plot", dest="dim", action="store_const", const=3, default=2)

    baseline = command.add_parser("baseline", help="reproduce baseline benchmarks")
    baseline.add_argument("keywords", help="any keyword of the baseline (e.g. model, dataset)", metavar="keyword", nargs="+")
    baseline.add_argument("--no-eval", help="turn off evaluation", dest="eval", action="store_false")
    baseline.add_argument("--gpu", help="overwrite the number of GPUs", type=int)
    baseline.add_argument("--cpu", help="overwrite the number of CPUs per GPU", type=int)

    list = command.add_parser("list", help="list available baselines")

    return parser


def load_config(config_file):

    def get_dataset(x):
        if not isinstance(x, str):
            return x
        result = re.match("<(\w+)\.(\w+)>", x)
        if result:
            dataset, key = result.groups()
            dataset = getattr(gv.dataset, dataset)
            file_name = getattr(dataset, key)
            return file_name
        else:
            return x

    with open(config_file, "r") as fin:
        cfg = EasyDict(yaml.safe_load(fin))
    cfg = gv.util.recursive_map(cfg, lambda x: gv.auto if x == "auto" else x)
    cfg = gv.util.recursive_map(cfg, get_dataset)
    if "optimizer" in cfg.build:
        cfg.build.optimizer = gv.optimizer.Optimizer(**cfg.build.optimizer)

    return cfg


def run_main(args):
    cfg = load_config(args.config)
    if args.gpu:
        cfg.resource.gpus = range(args.gpu)
    if args.cpu:
        cfg.resource.cpu_per_gpu = args.cpu

    app = gap.Application(cfg.application, **cfg.resource)
    app.load(**cfg.graph)
    app.build(**cfg.build)
    app.train(**cfg.train)
    if args.eval and "evaluate" in cfg:
        app.evaluate(**cfg.evaluate)
    if "save" in cfg:
        app.save(**cfg.save)


def visualize_main(args):

    def load_data(file_name):
        extension = os.path.splitext(file_name)[1]
        if extension == ".txt":
            data = np.loadtxt(file_name)
        elif extension == ".npy":
            data = np.load(file_name)
        else:
            raise ValueError("Can't solve file type `%s`" % extension)
        return data

    vectors = load_data(args.file)
    if args.label:
        labels = load_data(args.label)
    else:
        labels = None

    gv.init_logging(logging.WARNING)

    app = gap.VisualizationApplication(args.dim, [0])
    app.load(vectors=vectors, perplexity=args.perplexity)
    app.build()
    app.train()
    app.visualization(Y=labels, save_file=args.save)


def baseline_main(args):
    config_path = get_config_path()

    configs = []
    for path, dirs, files in os.walk(config_path):
        for file in files:
            file = os.path.join(path, file)
            match = True
            for keyword in args.keywords:
                # print("(^|_)%s(_|$)" % keyword)
                result = re.search("[/\_.]%s[/\_.]" % keyword, file)
                if not result:
                    match = False
                    break
            if match:
                configs.append(file)
    if len(configs) == 0:
        raise ValueError("Can't find a baseline with keywords: %s" % ", ".join(args.keywords))
    if len(configs) > 1:
        configs = sorted(configs)
        configs = [""] + [os.path.relpath(config, config_path) for config in configs]
        raise ValueError("Ambiguous keywords. Candidates are:%s" % "\n    ".join(configs))

    config = configs[0]
    print("running baseline: %s" % os.path.relpath(config, config_path))
    cfg = load_config(config)
    if args.gpu:
        cfg.resource.gpus = range(args.gpu)
    if args.cpu:
        cfg.resource.cpu_per_gpu = args.cpu

    app = gap.Application(cfg.application, **cfg.resource)
    app.load(**cfg.graph)
    app.build(**cfg.build)
    app.train(**cfg.train)
    if args.eval and "evaluate" in cfg:
        app.evaluate(**cfg.evaluate)
    if "save" in cfg:
        app.save(**cfg.save)


def list_main(args):
    config_path = get_config_path()

    print("list of baselines")
    print()
    indent = " " * 4
    count = 0
    for path, dirs, files in os.walk(config_path):
        path = os.path.relpath(path, config_path)
        depth = path.count("/")
        if path != ".":
            depth += 1
            print("%s%s" % (indent * depth, os.path.basename(path)))
        for file in sorted(files):
            print("%s%s" % (indent * (depth + 1), file))
        count += len(files)
        print()
    print("total: %d baselines" % count)


command = {
    "run": run_main,
    "visualize": visualize_main,
    "baseline": baseline_main,
    "list": list_main
}

def main():
    parser = get_parser()
    args = parser.parse_args()
    command[args.command](args)
    return 0