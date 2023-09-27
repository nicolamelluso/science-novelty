import numpy as np
import csv
import os
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

# Try to import cupy for GPU acceleration, fall back to numpy if not available
try:
    import cupy as xp
    print("Running on GPU")
except ImportError:
    import numpy as xp
    print("Running on CPU")

CHUNK_SIZE = 1000  # Adjust based on memory availability
N_JOBS = -1  # Use all available cores

def load_vectors_for_year(year, path_vectors):
    print(f'Reading {year}...')
    file_path = os.path.join(path_vectors, f"{year}_vectors.csv")
    data = xp.loadtxt(file_path, delimiter='\t', dtype=np.float32)
    papers_ids = data[:, 0].astype(xp.int64)
    vectors = data[:, 1:]
    return papers_ids, vectors

def calculate_similarity_for_chunk(chunk, prior_data):
    chunk_norm = chunk / xp.linalg.norm(chunk, axis=1, keepdims=True)
    prior_data_norm = prior_data / xp.linalg.norm(prior_data, axis=1, keepdims=True)
    similarities = xp.dot(chunk_norm, prior_data_norm.T)
    avg_dists = xp.mean(similarities, axis=1)
    max_dists = xp.max(similarities, axis=1)
    return avg_dists, max_dists

def calculate_avg_max_similarity(current_data, prior_data):
    results = Parallel(n_jobs=N_JOBS)(
        delayed(calculate_similarity_for_chunk)(current_data[i:i+CHUNK_SIZE], prior_data)
        for i in tqdm(range(0, len(current_data), CHUNK_SIZE))
    )
    avg_similarities = xp.concatenate([res[0] for res in results])
    max_similarities = xp.concatenate([res[1] for res in results])
    return avg_similarities, max_similarities

def initialize_output_file(output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['PaperId', 'cosine_max', 'cosine_avg'])

def save_to_csv(papers_ids, avg_similarities, max_similarities, output_path):
    with open(output_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for paper_id, avg_sim, max_sim in zip(papers_ids, avg_similarities, max_similarities):
            writer.writerow([paper_id, max_sim, avg_sim])

def calculate_similarities(start_year, end_year, input_dir, output_dir):
    output_path = os.path.join(output_dir, 'papers_cosine.csv')
    initialize_output_file(output_path)
    rolling_data = []
    years = range(start_year, end_year + 1)

    for year in tqdm(years):
        papers_ids, current_year_data = load_vectors_for_year(year, input_dir)
        rolling_data.append((year, current_year_data))
        rolling_data = [(y, data) for y, data in rolling_data if year - y < 6]

        if len(rolling_data) < 6:
            continue

        prior_data = xp.vstack([data for y, data in rolling_data if y != year])
        print('Calculating similarities for %d...' % (year))
        avg_year_similarities, max_year_similarities = calculate_avg_max_similarity(current_year_data, prior_data)
        save_to_csv(papers_ids, avg_year_similarities, max_year_similarities, output_path)

