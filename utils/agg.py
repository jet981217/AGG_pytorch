from typing import List

import torch


class agg(torch.nn.Module):
    def __init__(
        self,
        alpha: float,
        memory_len: int,
        token_ids_to_use: List[int],
        use_gpu: bool
    ):
        """Module to apply agg

        Args:
            alpha (float):
                Hyper-parameter alpha 
            memory_len (int):
                K step to track
            token_ids_to_use (List[int]):
                List of token id which is not [MASK], [CLS], ...
        """    
        self.__token_ids_to_use = token_ids_to_use
        self.__memory_cell = torch.Tensor(
                [[]*self.__token_ids_to_use]*memory_len
            ).to(
                "cuda" if use_gpu else "cpu"
        )
        self.__step = 0
        self.__alpha = alpha

        self.__less_rare_token_ids = []
        self.__very_rare_token_ids = []

    def dynamic_rare_token_grouping(self):
        ...

    def get_gate_mask(self):
        ...
    
    def step_agg(
        self,
        input_tokens: torch.Tensor
    ):
        self.__step += 1
        
