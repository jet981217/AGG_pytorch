"""Copyright 2023 by @jet981217. All rights reserved."""
from typing import Dict
import random

import torch
from transformers import (
    BertConfig,
    RobertaConfig,
)
import numpy as np

DATASET_ALIAS = {
    "wikitext": [
        "WikiText",
        "wikitext",
        "Wikitext",
        "wiki_text"
    ]
}

MODEL_ALIAS = {
    "vanilla-transformer-base": [
        "vanilla-transformer-base",
        "Vanilla-Transformer-base",
        "transformer-base",
        "Transformer-base",
    ]
}

MODEL_CONFIG_PATHS = {
    "vanilla-transformer-base": \
        "configs/model_configs/vanilla-transformer-base.json"
}

CONFIG_CLASSES = {
    "vanilla-transformer-base": BertConfig,
    "bert": BertConfig,
    "roberta": RobertaConfig,
}

TASKS = {
    "with_head": [
        "cls",
        "classification",
        "Classification",
        "lp",
        "linearprobe",
        "LinearProbe",
        "wef",
        "wordembeddingfreeze",
        "WordEmbeddingFreeze",
        "mlmcls",
        "mlmcls",
    ],
    "without_head": ["sim", "similarity", "Similarity"],
}

def set_seed(train_params: Dict):
    """Set random seed for every condition

    Args:
        train_params (Dict):
            Train params in train config dict
    """
    random.seed(train_params["seed"])
    np.random.seed(train_params["seed"])
    torch.manual_seed(train_params["seed"])
    if not train_params["no_cuda"] and torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_params["seed"])
