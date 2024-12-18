import csv
import collections
from typing import List, Tuple, Dict, Set
from tqdm import tqdm
import itertools as it


def calculate_new_ngram_combs(baseline_year: int, raw_papers_path: str, processed_papers_path: str) -> callable:
    """
    Prepare a function to calculate new words from the given papers data.
    
    Parameters:
    - baseline_year: The year to separate the baseline papers and the papers to be analyzed.
    - raw_papers_path: The path to the raw papers data CSV file.
    - processed_papers_path: The path to the processed papers data CSV file.
    
    Returns:
    - A function that calculates and returns new words and their counts.
    """
    
    # Create a baseline set of words from papers published before the baseline year
    baseline = set()
    with open(raw_papers_path, 'r', encoding='utf-8') as raw_reader,\
        open(processed_papers_path, 'r', encoding='utf-8') as processed_reader:
        
        csv_raw_reader = csv.reader(raw_reader, delimiter='\t', quotechar='"')
        csv_processed_reader = csv.reader(processed_reader, delimiter=',', quotechar='"')

        next(csv_raw_reader)
        next(csv_processed_reader)
        
        for line_raw, line_processed in zip(csv_raw_reader, csv_processed_reader):
            if int(line_raw[1].split('-')[0]) > baseline_year:
                continue
            
            text = set(line_processed[1].split() + line_processed[2].split())
        
            combs = list(it.combinations(text,2))
            combs = set([tuple(sorted(comb)) for comb in combs])

            baseline.update(combs)    
            
        print('Baseline built.')
            
    def new_ngram_comb_counter() -> Dict[str, Tuple[int, int]]:
        """
        Calculate new ngrams (words or phrases) that are not in the baseline.
        
        Returns:
        - A dictionary where keys are new words, and values are tuples containing the paper ID where the word first appeared
          and the number of times the word has been reused in other papers.
        """
        counter = collections.Counter()
        paperIds = collections.defaultdict(int)

        with open(processed_papers_path, 'r', encoding='utf-8') as reader:
            csv_reader = csv.reader(reader, delimiter=',', quotechar='"')
            next(csv_reader)

            for line in tqdm(csv_reader):
                paperID = int(line[0])
                text = set(line[1].split() + line[2].split())
                combs = list(it.combinations(text,2))
                combs = set([tuple(sorted(comb)) for comb in combs])

                for comb in combs:
                    if comb in baseline:
                        continue

                    if comb not in counter:
                        counter[comb] = 0
                        paperIds[comb] = paperID
                    else:
                        counter[comb] += 1
        
        # Prepare the final result, filtering out words that were only used once
        result = {comb: (paperIds[comb], count) for comb, count in counter.items() if count > 0}
        return result
    
    print('Done.')
    
    return new_ngram_comb_counter

