import csv
import torch
import numpy as np
from tqdm.notebook import tqdm

import sys
sys.path.insert(1, '../science_novelty/')

import embeddings

def process_embeddings(input_file, output_dir, storage='csv', chunk_size=50):
    # Load the embedding model
    print('Load the embedding model...')
    tokenizer, model = embeddings.load_model()

    # Move the model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("Model moved to GPU.")
    else:
        print("Using CPU.")

    # Count the number of papers
    print('Get the number of papers to process...')
    with open(input_file, 'r', encoding='utf-8') as file:
        line_count = sum(1 for line in file)
    total_papers = line_count - 1  # Subtract 1 for the header

    print('Processing...')
    current_year_vectors = []
    current_year = None

    with open(input_file, 'r', encoding='utf-8') as reader:
        csv_reader = csv.reader(reader, delimiter='\t', quotechar='"')
        next(csv_reader)  # Skip header

        for chunk_start in tqdm(range(0, total_papers, chunk_size)):
            chunk_data = [line for _, line in zip(range(chunk_size), csv_reader)]
            
            # Generate embeddings for the chunk
            for line in chunk_data:
                year = int(line[1].split('-')[0])
                
                if current_year is None:
                    current_year = year
                
                if year != current_year:
                    save_vectors(current_year, current_year_vectors, output_dir, storage)
                    current_year_vectors = []
                    current_year = year
                
                text = line[2] + line[3]
                
                vector = embeddings.get_embedding(text, tokenizer, model)
                
                current_year_vectors.append([line[0]] + list(vector))

        save_vectors(current_year, current_year_vectors, output_dir, storage)

def save_vectors(year, vectors, output_dir, storage):
    if storage == 'csv':
        with open(f'{output_dir}/{year}_vectors.csv', 'w', encoding='utf-8', newline='') as writer:
            csv_writer = csv.writer(writer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["PaperID"] + list(range(0, 768)))
            csv_writer.writerows(vectors)
    elif storage == 'numpy':
        np.save(f'{output_dir}/{year}_vectors.npy', np.array(vectors))
