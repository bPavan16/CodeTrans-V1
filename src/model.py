import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import List, Tuple
from tqdm import tqdm
import warnings

from dataset import CodeTranslationDataset, program_dataset

warnings.filterwarnings("ignore")


class CodeTranslator:
    def __init__(self, model_name="t5-small", device="cuda"):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    def train(
        self,
        train_java: List[str],
        train_python: List[str],
        val_java: List[str] = None,
        val_python: List[str] = None,
        batch_size: int = 8,
        epochs: int = 10,
        learning_rate: float = 2e-5,
    ):

        # Create datasets
        train_dataset = CodeTranslationDataset(train_java, train_python, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_java and val_python:
            val_dataset = CodeTranslationDataset(val_java, val_python, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix({"loss": loss.item()})

            avg_train_loss = total_loss / len(train_loader)
            print(f"\nAverage training loss: {avg_train_loss:.4f}")

            # Validation
            if val_java and val_python:
                val_loss = self.evaluate(val_loader)
                print(f"Validation loss: {val_loss:.4f}")

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                total_loss += outputs.loss.item()

        return total_loss / len(val_loader)

    def translate(self, java_code: str) -> str:
        self.model.eval()

        # Prepare input with task prefix
        input_text = f"translate Java to Python: {java_code}"
        inputs = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                num_beams=5,
                early_stopping=True,
            )

        # Decode output
        python_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return python_code


def get_training_data() -> Tuple[List[str], List[str]]:
    java_samples = program_dataset("Java")
    python_samples = program_dataset("Python")

    return java_samples, python_samples
