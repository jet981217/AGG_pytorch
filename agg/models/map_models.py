"""Module of classification head for every Ugly Classifier models"""
import json
from pathlib import Path
from typing import Dict, Union

import torch

from agg.models.vanialla_transformer.vanilla_transformer \
    import TransformerLM, TransformerMLM

from agg.utils.utils import MODEL_ALIAS, MODEL_CONFIG_PATHS, CONFIG_CLASSES

def map_model_config(
    model_name: str,
    model_config: Union[Path, str],
    train_config: Dict,
    is_pretrain: bool = False,
) -> any:
    """Map model config of LM

    Args:
        model_name (str): Type of LM to use
        model_config (str): Path of LM's config file
        train_config (Dict): Config dict of training
        is_pretrain (bool): method to check if
            it is pretrain mode. Default as False

    Returns:
        PretrainedConfig: Config class of PLM
    """
    return (
        CONFIG_CLASSES[model_name].from_pretrained(
            model_config,
            num_labels=len(list(train_config["id2label"].keys())),
            id2label=train_config["id2label"],
            label2id=train_config["label2id"],
        )
        if not is_pretrain
        else CONFIG_CLASSES[model_name].from_pretrained(
            model_config,
            num_labels=0,
            id2label={},
            label2id={},
        )
    )


def map_mlm_model(train_config: Dict) -> torch.nn.parallel.DataParallel:
    """Map a mlm model class from the train config dict

    Args:
        train_config (Dict): Config dict of train

    Raises:
        Exception: The model requested is not available

    Returns:
        torch.nn.parallel.DataParallel: DataParallel model
    """
    model = train_config["model"]
    
    if model in MODEL_ALIAS["vanilla-transformer-base"]:
        model_config = map_model_config(
            model_name="vanilla-transformer-base",
            model_config=MODEL_CONFIG_PATHS["vanilla-transformer-base"],
            train_config=train_config,
            is_pretrain=True,
        )
        model = TransformerLM(
            model_config=model_config
        )
        model_func = TransformerMLM
    else:
        raise Exception("Wrong model type!!!")

    model = model_func(
        LM=model,
        model_config=model_config,
    )

    if train_config["pretrained_weight"] != "":
        model.load_state_dict(
            torch.load(train_config["pretrained_weight"], map_location="cpu")
        )

    return torch.nn.parallel.DataParallel(model)
