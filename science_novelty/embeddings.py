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
    # Tokenize the text
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    
    # Move the tokenized input to the GPU if available
    if torch.cuda.is_available():
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach()
    
    # Move the embeddings to CPU for further operations
    embeddings = embeddings.cpu().numpy()

    return embeddings  # Added this line to return the embeddings




