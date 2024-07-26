import os

class Tokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = f.read().splitlines()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
    def encode_plus(self, text, add_special_tokens=True, max_length=None, return_token_type_ids=False, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'):
        tokens = text.split()
        input_ids = [self.token_to_id.get(token, self.token_to_id['<unk>']) for token in tokens]
        
        if add_special_tokens:
            input_ids = [self.token_to_id['<cls>']] + input_ids + [self.token_to_id['<sep>']]
        
        if max_length is not None:
            if truncation and len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
            if padding == 'max_length' and len(input_ids) < max_length:
                input_ids = input_ids + [self.token_to_id['<pad>']] * (max_length - len(input_ids))
        
        attention_mask = [1] * len(input_ids)
        if padding == 'max_length' and len(input_ids) < max_length:
            attention_mask += [0] * (max_length - len(input_ids))
        
        return {
            'input_ids': torch.tensor([input_ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long),
        }
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        with open(f"{save_directory}/vocab.txt", 'w') as f:
            f.write('\n'.join(self.vocab))
    
    @classmethod
    def from_pretrained(cls, save_directory):
        vocab_file = f"{save_directory}/vocab.txt"
        return cls(vocab_file)