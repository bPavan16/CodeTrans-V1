import pandas as pd
from torch.utils.data import Dataset

from typing import List
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


def program_dataset(language):
    dataset = pd.read_csv("data/data.csv")
    program_set = []

    for program in dataset[language]:
        program_set.append(program)

    return program_set


class CodeTranslationDataset(Dataset):
    def __init__(
        self, java_codes: List[str], python_codes: List[str], tokenizer, max_length=512
    ):
        self.java_codes = java_codes
        self.python_codes = python_codes
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.java_codes)

    def __getitem__(self, idx):
        java_code = self.java_codes[idx]
        python_code = self.python_codes[idx]

        # Add prefix for T5 to understand the task
        input_text = f"translate Java to Python: {java_code}"
        target_text = python_code

        # Tokenize inputs and outputs
        input_tokens = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        target_tokens = self.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": input_tokens["input_ids"].squeeze(),
            "attention_mask": input_tokens["attention_mask"].squeeze(),
            "labels": target_tokens["input_ids"].squeeze(),
        }
