import csv
import os
import torch
import numpy as np
from tqdm.notebook import tqdm

import sys
sys.path.insert(1, '../science_novelty/')

import embeddings

# Constants
PATH_OUTPUT = '../data/'
PATH_INPUT = '../data/raw/'
STORAGE = 'csv'
CHUNK_SIZE = 50
TOTAL_PAPERS = None

# Check if paths exist
if not os.path.exists(PATH_OUTPUT) or not os.path.exists(PATH_INPUT):
    raise Exception("Input or output path does not exist.")

def process_embeddings(input_file, output_dir, storage='csv', chunk_size=50):
    # Load the embedding model
    print('Loading the embedding model...')
    tokenizer, model = embeddings.load_model()

    # Move the model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Using {device.upper()}.")

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
    vectors_path = os.path.join(output_dir, 'vectors')
    os.makedirs(vectors_path, exist_ok=True)  # Ensure the directory exists

    file_path = os.path.join(vectors_path, f'{year}_vectors')

    if storage == 'csv':
        file_path += '.csv'
        mode = 'a' if os.path.exists(file_path) else 'w'
        with open(file_path, mode, encoding='utf-8', newline='') as writer:
            csv_writer = csv.writer(writer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if mode == 'w':
                print(f'Creating new file for year {year}...')
                csv_writer.writerow(["PaperID"] + [f"{i}" for i in range(len(vectors[0]) - 1)])  # Adjusted header format
            csv_writer.writerows(vectors)
    elif storage == 'numpy':
        file_path += '.npy'
        vectors = np.array([vec[1:] for vec in vectors])  # Exclude PaperID for numpy storage
        if os.path.exists(file_path):
            existing_vectors = np.load(file_path, allow_pickle=True)
            vectors = np.vstack((existing_vectors, vectors))
        np.save(file_path, vectors)
    else:
        raise ValueError("Unsupported storage format. Use 'csv' or 'numpy'.")