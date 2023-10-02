#!/usr/bin/env python
# coding: utf-8


from transformers import AutoTokenizer, AutoModel
import torch


# Move the model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_name = 'specter'):
    
    if 'specter' in model_name.lower():
        model = 'allenai/specter'
    elif 'scibert' in model_name.lower():
        model = 'allenai/scibert_scivocab_uncased'
    else:
        model = model_name
        
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    
    return tokenizer, model

def get_embedding(texts, tokenizer, model, max_length=512):
    inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings





