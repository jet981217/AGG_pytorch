"""Copyright 2023 by @jet981217. All rights reserved."""
import json
import itertools
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from transformers import AdamW
from transformers import logging as transformers_logging

from agg.utils.scheduler import CosineAnnealingWarmUpRestarts
from agg.engine.pretrain.evaler import EvalerMLM
from agg.utils.agg import AGG

logger = logging.getLogger(__name__)

transformers_logging.set_verbosity_error()

# pyright: reportGeneralTypeIssues=false
# pylint: disable=logging-fstring-interpolation, too-many-branches


def TrainerMLM(
    train_config: Dict,
    model: torch.nn.parallel.DataParallel,
    dataloaders: List[torch.utils.data.DataLoader],
    log_root: Union[Path, str],
    pretrain_type: str,
    token_ids_to_use: List[int]
) -> Tuple[int, float]:
    """Pretrain a model with mlm method

    Args:
        train_config (Dict):
            Config dict for training
        model (torch.nn.parallel.DataParallel):
            Model to pretrain
        dataloaders (List[torch.utils.data.DataLoader]):
            Train and val dataloader
        log_root (Union[Path, str]):
            Root dir to save log and checkpoints

    Returns:
        Tuple[int, float]:
            Global step and avg train loss
    """
    token_controller = AGG(
        device="cuda" if torch.cuda.is_available() else "cpu",
        alpha=train_config["agg_params"]["alpha"],
        memory_len=train_config["agg_params"]["memory_len"],
        vocab_size=model.module.LM.embeddings.weight.size()[0],
        token_ids_to_use=token_ids_to_use
    ) if pretrain_type == "agg" else AGG

    train_config["result"] = {
        "val_best_acc": 0.0,
        "val_best_token_acc": 0.0,
        "val_best_loss": float("inf"),
    }
    train_config["cur_patience"] = train_config["train_params"]["patience"]

    prev_best = 0.0

    train_dataloader = dataloaders[0]

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": train_config["train_params"]["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=0,
        eps=train_config["train_params"]["adam_epsilon"],
    )
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,
        T_0=train_config["train_params"]["T_0"] * len(train_dataloader),
        T_mult=train_config["train_params"]["T_mult"],
        eta_max=train_config["train_params"]["max_learning_rate"],
        T_up=int(
            train_config["train_params"]["T_0"]
            * len(train_dataloader)
            * train_config["train_params"]["warmup_proportion"]
        ),
        gamma=train_config["train_params"]["scheduler_gamma"],
    )
    if (
        os.path.exists(train_config["optimizer_path"])
        and os.path.exists(train_config["scheduler_path"])
        and os.path.isfile(train_config["optimizer_path"])
        and os.path.isfile(train_config["scheduler_path"])
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(train_config["optimizer_path"]))
        scheduler.load_state_dict(torch.load(train_config["scheduler_path"]))

    # Train!
    global_step = 0
    tr_loss = 0.0

    model.zero_grad()

    for epoch in range(int(train_config["train_params"]["num_train_epochs"])):
        print(f"Training epoch {epoch}...")
        
        epoch_iterator = train_dataloader
        mb = epoch_iterator

        tqdm_logger = tqdm(mb, total=len(mb))

        train_config["cur_epoch"] = epoch + 1

        for step, batch in enumerate(tqdm_logger):
            train_config["cur_step"] = global_step

            model.train()
            outputs = []

            batch_id, batch_mask, labels, masked_idxs = batch

            token_controller.step_agg(
                input_tokens_batch=batch_id
            )
            
            embedding_matrix_mask = [
                token_controller.get_gate_mask(
                    target_tokens = [i.item() for i in torch.nonzero(masked_idx)]
                )
                for masked_idx in masked_idxs
                if masked_idx.count_nonzero() > 0
            ]
            embedding_matrix_mask = torch.mean(
                torch.stack(embedding_matrix_mask, dim=0),
                dim=0
            ) if len(embedding_matrix_mask) > 0 else None

            detach_ratios = embedding_matrix_mask.view(-1, 1)
            model.module.LM.embeddings.weight = torch.nn.Parameter(
                detach_ratios * model.module.LM.embeddings.weight + \
                    (1 - detach_ratios) * model.module.LM.embeddings.weight.detach()
            )

            inputs = {
                "input_ids": batch_id.to(train_config["device"]),
                "attention_mask": batch_mask.to(train_config["device"]),
            }

            outputs = model(**inputs)
            criterion = torch.nn.CrossEntropyLoss()

            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                labels.to(train_config["device"]).reshape(-1),
            )

            if train_config["train_params"]["gradient_accumulation_steps"] > 1:
                loss = (
                    loss
                    / train_config["train_params"][
                        "gradient_accumulation_steps"
                    ]
                )

            loss.backward()

            tr_loss += loss.item()
            model.module.LM.embeddings.weight = \
                torch.nn.Parameter(
                    model.module.LM.embeddings.weight.detach().requires_grad_(True)
                )

            if (step + 1) % train_config["train_params"][
                "gradient_accumulation_steps"
            ] == 0 or (
                len(train_dataloader)
                <= train_config["train_params"]["gradient_accumulation_steps"]
                and (step + 1) == len(train_dataloader)
            ):
                clip_grad_norm_(
                    model.parameters(),
                    train_config["train_params"]["max_grad_norm"],
                )

                optimizer.step()
                scheduler.step()
                model.zero_grad()

                global_step += 1

            if (
                train_config["train_params"]["logging_steps"] > 0
                and (global_step + 1)
                % train_config["train_params"]["logging_steps"]
                == 0
            ):
                prev_best = train_config["result"]["val_best_token_acc"]

                val_loss, val_acc, val_token_acc = EvalerMLM(
                    model=model,
                    dataloader=dataloaders[1],
                    train_config=train_config,
                )

                if val_token_acc > train_config["result"]["val_best_token_acc"]:
                    train_config["result"]["val_best_loss"] = val_loss
                    train_config["result"]["val_best_acc"] = val_acc.item()
                    train_config["result"]["val_best_token_acc"] = val_token_acc

                tqdm_logger.set_postfix(
                    {
                        "cur_val_loss": val_loss,
                        "val_best_loss": train_config["result"][
                            "val_best_loss"
                        ],
                        "cur_val_acc": val_acc.item(),
                        "val_best_acc": train_config["result"]["val_best_acc"],
                        "cur_val_token_acc": val_token_acc,
                        "val_best_token_acc": train_config["result"][
                            "val_best_token_acc"
                        ],
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "cur_patience": train_config["cur_patience"],
                    }
                )
            if (
                train_config["train_params"]["logging_steps"] > 0
                and (global_step + 1)
                % train_config["train_params"]["logging_steps"]
                == 0
            ):
                # Save model checkpoint
                output_dir = os.path.join(log_root, "last")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )
                model_to_save.LM.save_pretrained(output_dir)

                with open(
                    os.path.join(output_dir, "training_args_logs.json"), "w"
                ) as training_args_logs:
                    json.dump(train_config, training_args_logs)
                logger.info(f"Saving model checkpoint to {output_dir}")

                if train_config["train_params"]["save_optimizer"]:
                    torch.save(
                        optimizer.state_dict(),
                        os.path.join(output_dir, "optimizer.pt"),
                    )
                    torch.save(
                        scheduler.state_dict(),
                        os.path.join(output_dir, "scheduler.pt"),
                    )
                    logger.info(
                        f"Saving optimizer and scheduler states to {output_dir}"
                    )

                if train_config["result"]["val_best_token_acc"] > prev_best:
                    # Save model checkpoint
                    output_dir = os.path.join(log_root, "best")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )
                    model_to_save.LM.save_pretrained(output_dir)

                    with open(
                        os.path.join(output_dir, "training_args_logs.json"), "w"
                    ) as training_args_logs:
                        json.dump(train_config, training_args_logs)
                    logger.info(f"Saving model checkpoint to {output_dir}")

                    if train_config["train_params"]["save_optimizer"]:
                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(output_dir, "optimizer.pt"),
                        )
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(output_dir, "scheduler.pt"),
                        )
                        logger.info(
                            "Saving optimizer and "
                            f"scheduler states to {output_dir}"
                        )
                    train_config["cur_patience"] = train_config["train_params"][
                        "patience"
                    ]
                else:
                    train_config["cur_patience"] -= 1
                    if train_config["cur_patience"] < 0:
                        return global_step, tr_loss / global_step

    return global_step, tr_loss / global_step
