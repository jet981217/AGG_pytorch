"""Moudle to map dataloader(s) with config file"""
from typing import Dict, List, Tuple

from torch.utils.data import DataLoader

from agg.data.tokenizer import map_tokenizer
from agg.data.dataset import WikiTextMLM
from agg.utils.utils import DATASET_ALIAS


def prepare_mlm_dataloader(
    pretrain_config: Dict,
) -> List[DataLoader]:
    """Method to map mlm dataloaders when training

    Args:
        pretrain_config (Dict): Config dict for training

    Returns:
        Tuple[List[int], List[DataLoader]]: train & val Dataloader
    """
    dataloaders = []

    tokenizer = map_tokenizer(pretrain_config)

    if pretrain_config["dataset"] in DATASET_ALIAS["wikitext"]:
        dataset_class = WikiTextMLM

    train_dataset = dataset_class(
        tokenizer=tokenizer,
        train_config=pretrain_config,
        mode="train",
    )
    val_dataset = dataset_class(
        tokenizer=tokenizer,
        train_config=pretrain_config,
        mode="validation",
    )
    test_dataset = dataset_class(
        tokenizer=tokenizer,
        train_config=pretrain_config,
        mode="test",
    )

    dataloaders.extend(
        [
            DataLoader(
                train_dataset,
                batch_size=pretrain_config["batch_size"]["train"],
                num_workers=pretrain_config["num_workers"]
                if "num_workers" in pretrain_config
                else 0,
            ),
            DataLoader(
                val_dataset,
                batch_size=pretrain_config["batch_size"]["val"],
                num_workers=pretrain_config["num_workers"]
                if "num_workers" in pretrain_config
                else 0,
            ),
            DataLoader(
                test_dataset,
                batch_size=pretrain_config["batch_size"]["test"],
                num_workers=pretrain_config["num_workers"]
                if "num_workers" in pretrain_config
                else 0,
            ),
        ]
    )

    return dataloaders
