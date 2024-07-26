import torch
from tokenizer import Tokenizer
from model import Model, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward
from safetensors.torch import load_file

def preprocess_text(text, tokenizer, max_length):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return encoding["input_ids"].flatten(), encoding["attention_mask"].flatten()

def load_and_predict(model_path, tokenizer_path, text, max_length):
    tokenizer = Tokenizer.from_pretrained(tokenizer_path)
    
    model = Model(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        max_seq_length=max_length
    )
    model.load_state_dict(load_file(model_path))
    model.eval()

    input_ids, attention_mask = preprocess_text(text, tokenizer, max_length)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    
    src_key_padding_mask = (attention_mask == 0)

    with torch.no_grad():
        output = model(input_ids.transpose(0, 1), src_key_padding_mask)
        prediction = torch.sigmoid(output).item()
    
    return f'Input text: {text}\nPrediction probability: {prediction:.4f}\nPredicted class: {"positive" if prediction > 0.5 else "negative"}'

print(load_and_predict("model.safetensors", "tokenizer", "happy birthday", max_length=128))
