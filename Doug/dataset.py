import torch
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from model import tokenizer, max_seq_length

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float)
        }


dataset = load_dataset("nroggendorff/doug")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    dataset["train"]["text"], dataset["train"]["label"], test_size=0.2, random_state=42
)

train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_seq_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_seq_length)