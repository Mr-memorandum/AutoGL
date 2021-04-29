"""
Auto graph classification using cross validation methods proposed in
paper `A Fair Comparison of Graph Neural Networks for Graph Classification`
"""

import sys
import os
import os.path as osp
import random
import torch
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

sys.path.append("../")

from autogl.datasets import build_dataset_from_name, utils
from autogl.solver import AutoGraphClassifier
from autogl.module import Acc

import os.path as osp

import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from tqdm import tqdm
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


if __name__ == "__main__":
    parser = ArgumentParser(
        "auto graph classification", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default="mutag",
        type=str,
        help="graph classification dataset",
        choices=["mutag", "imdb-b", "proteins"],
    )
    parser.add_argument(
        "--configs", default="../configs/graphclf_full.yml", help="config files"
    )
    parser.add_argument("--device", type=int, default=0, help="device to run on")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--folds", type=int, default=10, help="fold number")
    parser.add_argument("--num_workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--train_split", type=float, default=0.9, help="the train split when fitting")

    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    seed = args.seed
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("begin processing dataset", args.dataset, "into", args.folds, "folds.")
    dataset = build_dataset_from_name(args.dataset)
    if args.dataset.startswith("imdb"):
        dataset = build_dataset_from_name(args.dataset)
        from autogl.module.feature.generators import PYGOneHotDegree

        # get max degree
        from torch_geometric.utils import degree

        max_degree = 0
        for data in dataset:
            deg_max = int(degree(data.edge_index[0], data.num_nodes).max().item())
            max_degree = max(max_degree, deg_max)
        dataset = PYGOneHotDegree(max_degree).fit_transform(dataset, inplace=False)
    elif args.dataset == "collab":
        dataset = build_dataset_from_name(args.dataset)
        from autogl.module.feature.auto_feature import Onlyconst

        dataset = Onlyconst().fit_transform(dataset, inplace=False)
    if not os.path.exists(f'{args.dataset}_id.json'):
        idx_list = list(range(len(dataset)))
        random.shuffle(idx_list)
        json.dump(idx_list, open(f'{args.dataset}_id.json', 'w'))
    dataset = dataset[json.load(open(f'{args.dataset}_id.json'))]

    accs = []
    CHUNK = len(dataset) // args.folds
    for fold in range(args.folds):
        print("evaluating on fold number:", fold)

        if fold < args.folds - 1:
            test_dataset = dataset[fold * CHUNK:fold * CHUNK + CHUNK]
            idxs = list(range(fold * CHUNK)) + list(range(fold * CHUNK + CHUNK, len(dataset)))
            train_dataset = dataset[idxs]
        else:
            test_dataset = dataset[fold * args.folds - 1:]
            train_dataset = dataset[:fold * args.folds - 1]
        autoClassifier = AutoGraphClassifier.from_config(args.configs)
        autoClassifier.num_workers = args.num_workers

        autoClassifier.fit(
            train_dataset,
            train_split=args.train_split,
            val_split=1 - args.train_split,
            seed=args.seed,
            evaluation_method=[Acc],
        )
        test_dataset.test_split = test_dataset
        predict_result = autoClassifier.predict_proba(test_dataset, mask="test")
        label = [data.y.item() for data in test_dataset]
        acc = Acc.evaluate(
            predict_result, label
        )
        print("test acc %.4f" % (acc))
        accs.append(acc)
    print("Average acc on", args.dataset, ":", np.mean(accs), "~", np.std(accs))
