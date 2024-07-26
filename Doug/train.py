import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from safetensors.torch import save_file

from tokenizer import Tokenizer
from model import Model, vocab_size, d_model, dim_feedforward, max_seq_length, nhead, num_encoder_layers

import torch
from sklearn.model_selection import train_test_split
from datasets import load_dataset

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

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        
        src_key_padding_mask = (attention_mask == 0)
        output = model(input_ids.transpose(0, 1), src_key_padding_mask)
        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            src_key_padding_mask = (attention_mask == 0)
            output = model(input_ids.transpose(0, 1), src_key_padding_mask)
            
            loss = criterion(output, labels)
            total_loss += loss.item()

            predicted = (output > 0.5).float()
            correct += (predicted == labels).sum().item()

    return total_loss / len(val_loader), correct / len(val_loader.dataset)

dataset = load_dataset("nroggendorff/doug")

model = Model(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length)

def tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

unique_tokens = set()
for example in dataset['train']:
    tokens = tokenize(example['text'])
    unique_tokens.update(tokens)

vocab = ['<pad>', '<cls>', '<sep>', '<unk>'] + sorted(unique_tokens)

vocab_file_path = 'vocab.txt'
with open(vocab_file_path, 'w') as f:
    f.write('\n'.join(vocab))
    
train_texts, val_texts, train_labels, val_labels = train_test_split(
    dataset["train"]["text"], dataset["train"]["label"], test_size=0.2, random_state=42
)

tokenizer = Tokenizer(vocab_file='vocab.txt')

train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_seq_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_seq_length)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=1e-5)

num_epochs = 8

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    print(f'Epoch: {epoch+1}')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Val Loss: {val_loss:.4f}')
    print(f'Val Accuracy: {val_acc:.4f}')

model_state_dict = model.state_dict()
save_file(model_state_dict, "model.safetensors")

tokenizer.save_pretrained("tokenizer")