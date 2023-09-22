#!/usr/bin/env python
# coding: utf-8


from transformers import AutoTokenizer, AutoModel
import torch

def load_model(model_name = 'specter'):
    
    if 'specter' in model_name.lower():
        model = 'allenai/specter'
        
    elif 'scibert' in model_name.lower():
        model = 'allenai/scibert_scivocab_uncased'
        
    else:
        model_name == model
        
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    
    return tokenizer, model

def get_embedding(text, tokenizer, model, max_length=512):
    # Encode the text with a specified max_length
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

    # Compute the embedding
    with torch.no_grad():
        embedding = model(**inputs)['pooler_output']

    # Convert the tensor to a numpy array and then flatten it to get a single vector
    embedding = embedding.numpy().flatten()
    
    return embedding


