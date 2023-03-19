"""MLM dataset class for agg"""
import json
import random
from typing import Dict, Optional, Tuple
import re

from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer

# pyright: reportGeneralTypeIssues=false


class WikiTextMLM(Dataset):
    """MLM dataset class for UglyTitle"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        train_config: Dict,
        max_seq_length: int = 128,
        mode: Optional[str] = "train",
    ) -> None:
        """MLM Agg dataset init function

        Args:
            tokenizer (PreTrainedTokenizer): Input tokenizer
            train_config (Dict): Config dict for training
            max_seq_length (int, optional):
                Max seq length for input ids. Defaults to 128.
            mode (Optional[str], optional):
                Type of mode to use UglyTitle. Defaults to "train".
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.__max_seq_length = max_seq_length

        self.__features = WikiTextMLM.get_dataset(mode=mode)

        if mode == "train":
            random.shuffle(self.__features)

    def encode(self, text: str) -> BatchEncoding:
        """Encode a sentence with tokenizer

        Args:
            text (str): Sentence to encode

        Returns:
            BatchEncoding: Tokenized encoding
        """
        return self.tokenizer.encode_plus(
            text=text,
            max_length=self.__max_seq_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        """Get length of dataset

        Returns:
            int: The length of dataset
        """
        return len(self.__features)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """Get member for UglyTitle each iteration

        Args:
            i (int): index for iteration

        Raises:
            Exception: When the task you requested is not available

        Returns:
            Tuple[torch.Tensor]:
                Tensorized sentence and label
        """
        encoded = self.encode(self.__features[i]["text"])

        label = torch.Tensor(encoded["input_ids"]).squeeze().clone().detach()

        masked_sentence, masked_idxs = WikiTextMLM.mask_sentence(
            sequence=encoded["input_ids"].squeeze(),
            train_config=self.train_config,
            tokenizer=self.tokenizer,
            padding_idx=int(
                torch.nonzero(encoded["attention_mask"].squeeze())[-1][0] + 1
            ),
        )

        return (
            masked_sentence,
            torch.Tensor(encoded["attention_mask"]).squeeze(),
            label,
            torch.tensor(
                [
                    1 if i in masked_idxs else 0 for i in range(
                        self.tokenizer.vocab_size
                    )
                ], dtype=torch.float32
            )
        )

    @staticmethod
    def mask_sentence(
        sequence: torch.Tensor,
        train_config: Dict,
        tokenizer: PreTrainedTokenizer,
        padding_idx: int,
    ) -> torch.Tensor:
        """Mask a sentence for MLM

        Args:
            sequence (torch.Tensor):
                Tensor of sequnce to make as MLM form
            train_config (Dict):
                Config dict for training
            tokenizer (PreTrainedTokenizer):
                Tokenizer
            padding_idx (int):
                Idx where padding starts

        Returns:
            torch.Tensor: Tensor of sequence fit as MLM form
        """
        return_idx = []

        if padding_idx <= 2:
            return sequence, return_idx

        mask_prob = train_config["mask_prob"]

        applied_idx = random.sample(
            range(1, padding_idx), int(padding_idx * mask_prob) + 1
        )

        mask_idxs = applied_idx[: int(0.8 * len(applied_idx))]
        replace_token_idxs = applied_idx[int(0.9 * len(applied_idx)) :]

        return_idx.extend(sequence[mask_idxs])
        return_idx.extend(sequence[replace_token_idxs])

        sequence[mask_idxs] = 103
        for replace_token_idx in replace_token_idxs:
            sequence[replace_token_idx] = WikiTextMLM.get_replace_token(
                token_id=replace_token_idx, tokenizer=tokenizer
            )

        return sequence, return_idx

    @staticmethod
    def get_replace_token(token_id: int, tokenizer: PreTrainedTokenizer) -> int:
        """Get random idx for replacement

        Args:
            token_id (int):
                Original token id
            tokenizer (PreTrainedTokenizer):
                Tokenizer

        Returns:
            int: New token ID for replacement
        """
        return random.choice(
            list(set(range(999, tokenizer.vocab_size)) - set([token_id]))
        )

    @staticmethod
    def get_dataset(
        mode: str
    ):
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=mode)

        def preprocess(text):
            text = re.sub(r'=+\s.*?\s=+', '', text)
            text = re.sub(r'\n+', '\n', text)
            lines = [line for line in text.split('\n') if line.strip()]
            return '\n'.join(lines)

        dataset = dataset.map(lambda x: {'text': preprocess(x['text'])})
        dataset = dataset.filter(lambda x: len(x['text']) > 0)

        return list(dataset)
