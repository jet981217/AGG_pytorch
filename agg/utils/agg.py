"""Copyright 2023 by @jet981217. All rights reserved."""
from typing import List

import torch


class AGG:
    def __init__(
        self,
        device: str,
        alpha: float,
        memory_len: int,
        vocab_size: int,
        token_ids_to_use: List[int]
    ) -> None:
        """Adaptive gradient gating class

        Args:
            device (str):
                Device to handle tensors.
            alpha (float):
                Hyper-parameter to decide rare tokens.
            memory_len (int):
                Steps to log.
            vocab_size (int):
                Total size of vocab for tokenizer.
            token_ids_to_use (List[int]):
                Token ids which contains real word.
                [MASK], [CLS] etc will be cannot be part of this list
        """    
        self.__device = device
        self.__token_ids_to_use = token_ids_to_use

        self.__memory_cell = torch.Tensor(
            [[0]*vocab_size]*memory_len
        ).to(device)
        self.__appearance_rate = torch.ones_like(
            self.__memory_cell[0]
        ).to(device)

        self.__step = 0
        self.__alpha = alpha


    def dynamic_rare_token_grouping(self) -> None:
        """Method to dynamically group rare tokens for each step"""        
        boundary = self.__step \
            if self.__step < len(self.__memory_cell) else None
        memory_mean = torch.mean(self.__memory_cell[:boundary], dim=0)
        
        self.__appearance_rate[self.__token_ids_to_use] = \
            memory_mean[self.__token_ids_to_use] / \
            torch.sum(memory_mean[self.__token_ids_to_use])
        
        rare_tokens = [
            [idx, rate] for idx, rate in enumerate(self.__appearance_rate)
                if rate < self.__alpha and idx in self.__token_ids_to_use
        ]

        mean_apperance_rare = torch.mean(
            torch.Tensor(
                [rare_token_info[1] for rare_token_info in rare_tokens]
            )
        ).to(self.__device)

        self.__rare_tokens = [
            rare_token_info[0] for rare_token_info in rare_tokens
        ]
        self.__g1_gate_vector = torch.ones_like(self.__appearance_rate)
        self.__g1_gate_vector[self.__rare_tokens] = self.__appearance_rate[
            self.__rare_tokens
        ]
        self.__very_rare_token_ids = \
            torch.where(self.__appearance_rate / mean_apperance_rare<1)[0].tolist()
        self.__g2_gate_vector = torch.ones_like(self.__appearance_rate)
        self.__g2_gate_vector[
            self.__very_rare_token_ids
        ] = self.__appearance_rate[
            self.__very_rare_token_ids
        ] / mean_apperance_rare


    def step_agg(
        self,
        input_tokens_batch: torch.Tensor
    ) -> None:
        """Step agg

        Args:
            input_tokens_batch (torch.Tensor):
                Input tokens for a batch(step).
        """
        self.__memory_cell[
            self.__step % len(self.__memory_cell)
        ] = torch.Tensor([
            torch.numel(input_tokens_batch[input_tokens_batch==idx])
            for idx in range(len(self.__memory_cell[0]))
        ])
        self.__step += 1
        self.dynamic_rare_token_grouping()


    def get_gate_mask(self, target_tokens: List[int]) -> torch.Tensor:
        """Get gating mask for each case

        Args:
            target_token (int): _description_

        Returns:
            torch.Tensor: _description_
        """
        vectors = [
            self.__g2_gate_vector
                if target_token in self.__rare_tokens
                else self.__g1_gate_vector
            for target_token in target_tokens
        ]
        return torch.mean(torch.stack(vectors, dim=0), dim=0)
