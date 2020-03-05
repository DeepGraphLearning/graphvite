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
import glob
import yaml
import shutil
import logging
import argparse
from easydict import EasyDict

import numpy as np

import graphvite as gv
import graphvite.application as gap


def get_config_path():
    candidate_paths = [
        os.path.realpath(os.path.join(gv.package_path, "config")),
        os.path.realpath(os.path.join(gv.package_path, "../../config"))
    ]
    for config_path in candidate_paths:
        if os.path.isdir(config_path):
            return config_path
    raise IOError("Can't find configuration directory. Did you install GraphVite correctly?")


def get_parser():
    parser = argparse.ArgumentParser(description="GraphVite command line executor v%s" % gv.__version__)
    command = parser.add_subparsers(metavar="command", dest="command")
    command.required = True

    new = command.add_parser("new", help="create a new configuration file")
    new.add_argument("application", help="name of the application (e.g. graph)", nargs="+")
    new.add_argument("--file", help="yaml file to save")

    run = command.add_parser("run", help="run from configuration file")
    run.add_argument("config", help="yaml configuration file")
    run.add_argument("--no-eval", help="turn off evaluation", dest="eval", action="store_false")
    run.add_argument("--gpu", help="override the number of GPUs", type=int)
    run.add_argument("--cpu", help="override the number of CPUs per GPU", type=int)
    run.add_argument("--epoch", help="override the number of epochs", type=int)

    visualize = command.add_parser("visualize", help="visualize high-dimensional vectors")
    visualize.add_argument("file", help="data file (numpy dump or txt)")
    visualize.add_argument("--label", help="label file (numpy dump or txt)")
    visualize.add_argument("--save", help="png or pdf file to save")
    visualize.add_argument("--perplexity", help="perplexity for the neighborhood", type=float, default=30)
    visualize.add_argument("--3d", help="3d plot", dest="dim", action="store_const", const=3, default=2)

    baseline = command.add_parser("baseline", help="reproduce baseline benchmarks")
    baseline.add_argument("keywords", help="any keyword of the baseline (e.g. model, dataset)", metavar="keyword",
                          nargs="+")
    baseline.add_argument("--no-eval", help="turn off evaluation", dest="eval", action="store_false")
    baseline.add_argument("--gpu", help="overwrite the number of GPUs", type=int)
    baseline.add_argument("--cpu", help="overwrite the number of CPUs per GPU", type=int)
    baseline.add_argument("--epoch", help="override the number of epochs", type=int)

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
    if "vectors" in cfg.graph:
        if isinstance(cfg.graph.vectors, str) and cfg.graph.vectors.endswith(".npy"):
            cfg.graph.vectors = np.load(cfg.graph.vectors)

    return cfg


def new_main(args):
    config_path = get_config_path()
    template_path = os.path.join(config_path, "template")
    if not os.path.isdir(template_path):
        raise IOError("Can't find template configuration directory. Did you install GraphVite correctly?")

    config = "_".join(args.application) + ".yaml"
    template = os.path.join(template_path, config)
    if args.file:
        config = args.file
    if os.path.isfile(template):
        if os.path.exists(config):
            answer = None
            while answer not in ["y", "n"]:
                answer = input("File `%s` exists. Overwrite? (y/n)" % config)
            if answer == "n":
                return
        shutil.copyfile(template, config)
        print("A configuration template has been written into `%s`." % config)
    else:
        templates = glob.glob(os.path.join(template_path, "*.yaml"))
        templates = sorted(templates)
        applications = [""]
        for template in templates:
            application = os.path.splitext(os.path.basename(template))[0]
            application = application.replace("_", " ")
            applications.append(application)
        raise ValueError("Can't find a configuration template for `%s`. Available applications are %s"
                         % (" ".join(args.application), "\n    ".join(applications)))


def run_main(args):
    cfg = load_config(args.config)
    if args.gpu is not None:
        cfg.resource.gpus = range(args.gpu)
    if args.cpu is not None:
        cfg.resource.cpu_per_gpu = args.cpu
    if args.epoch is not None:
        cfg.train.num_epoch = args.epoch

    app = gap.Application(cfg.application, **cfg.resource)
    if "format" in cfg:
        app.set_format(**cfg.format)
    app.load(**cfg.graph)
    app.build(**cfg.build)
    if "load" in cfg:
        app.load_model(**cfg.load)
    app.train(**cfg.train)
    if args.eval and "evaluate" in cfg:
        if isinstance(cfg.evaluate, dict):
            cfg.evaluate = [cfg.evaluate]
        for evaluation in cfg.evaluate:
            app.evaluate(**evaluation)
    if "save" in cfg:
        app.save_model(**cfg.save)


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
                result = re.search(r"[/\\_.]%s[/\\_.]" % keyword, file)
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
    if args.gpu is not None:
        cfg.resource.gpus = range(args.gpu)
    if args.cpu is not None:
        cfg.resource.cpu_per_gpu = args.cpu
    if args.epoch is not None:
        cfg.train.num_epoch = args.epoch

    app = gap.Application(cfg.application, **cfg.resource)
    app.load(**cfg.graph)
    app.build(**cfg.build)
    if "load" in cfg:
        app.load_model(**cfg.load)
    app.train(**cfg.train)
    if args.eval and "evaluate" in cfg:
        if isinstance(cfg.evaluate, dict):
            cfg.evaluate = [cfg.evaluate]
        for evaluation in cfg.evaluate:
            app.evaluate(**evaluation)
    if "save" in cfg:
        app.save_model(**cfg.save)


def list_main(args):
    config_path = get_config_path()

    print("list of baselines")
    print()
    indent = " " * 4
    count = 0
    for path, dirs, files in os.walk(config_path):
        path = os.path.relpath(path, config_path)
        if path == "template":
            continue
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
    "new": new_main,
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