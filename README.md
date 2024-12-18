# Measuring Novel Scientific Ideas and their Impact in Publication Text using Natural Language Processing

## Citation

If you use the code from this repository, please cite the following paper: 
 > *Arts S., Melluso N., Veugelers R. (2025). Beyond Citations: Measuring Novel Scientific Ideas and their Impact in Publication Text. Forthcoming at the _Review of Economics and Statistics_. Preprint available at: https://doi.org/10.48550/arXiv.2309.16437*

## Overview

This repository is dedicated to the assessement of novelty and its impact of scientific publications, employing Python scripts and Jupyter notebooks. It is designed with a dual purpose:

- **Reproduce Results**:
  - Replicate the findings of the original paper, which analyzes data from OpenAlex, encompassing a comprehensive collection of papers from 1666 to 2023. The data can be accessed here: [https://doi.org/10.5281/zenodo.8283352](https://doi.org/10.5281/zenodo.8283352).
- **Custom Analysis**:
  - Enable users to apply the analysis, including preprocessing and metrics calculation, to a tailored set of papers for individual research needs.

![Science Novelty Schema](https://github.com/nicolamelluso/science-novelty/blob/main/data/ScienceNovelty-schema.png)
 

## Quick Tutorial

One of the primary and useful application of this repository is the pre-processing of text. Given a raw text, using the script `preprocessing.py` it is possible to get the cleaned words and noun phrases. This "processed" output can be used as input to measure the novelty of a paper or to trace the reuse of the constituent words or noun phrases. The `preprocessing.py` script strictly follows the procedure described in our [Zenodo](https://doi.org/10.5281/zenodo.8283352) repository, including stopwords posted in the repository. The `preprocessing.py` script can be improved. If you have any feedback please contact me [nicolamelluso@gmail.com](nicolamelluso@gmail.com).

This is a simple illustration on how to use the processing code:

```python
## Import the script
import preprocessing

text = "Specific enzymatic amplification of DNA in vitro: the polymerase chain reaction"

# Processed words
processed_words = preprocessing.process_text(text.lower(), 'words')

# Processed noun phrases
processed_phrases = preprocessing.process_text(text.lower(), 'phrases')

print("Processed Words:", processed_words)
print("Processed Phrases:", processed_phrases)
```

### Example output:

```text
Processed Words: ['specific', 'enzymatic', 'amplification', 'dna', 'vitro', 'polymerase', 'chain', 'reaction']

Processed Phrases: ['specific enzymatic amplification', 'dna', 'polymerase chain reaction']
```

## Preliminary tips for reproducibility
This repository demonstrates how to compute the metrics described in the associated paper. It is important to note that the original computations were executed on a server with approximately 500GB of RAM. Running the code as-is on less powerful machines is not recommended. Instead, we suggest optimizing the code to enable parallelized executions for better efficiency.

The primary goal of this repository is to illustrate the conceptual framework behind each metric rather than provide highly optimized, production-ready implementations. To achieve better reproducibility and scalability, we recommend utilizing more advanced computational resources. For example:

- Google BigQuery: Ideal for computing text-based metrics efficiently at scale.
- ChromaDB: Suitable for handling operations involving embeddings.
- 
We welcome any feedback or suggestions for improving the code to enhance its reproducibility and usability. If you have ideas for integrating optimized Google BigQuery queries or other techniques that streamline the identification and calculation of the metrics, please let us know.

For assistance or suggestions, feel free to contact Nicola Melluso at [nicolamelluso@gmail.com](nicolamelluso@gmail.com).

## Structure

The methodology is systematically organized into the following segments:
1. **Data Collection**
2. **Preprocessing**
3. **Text Embeddings**
4. **Semantic Distance**
5. **New Words**
6. **New Phrases**
7. **New Word Combinations**
8. **New Phrase Combinations**

Each segment is integral for extracting text-based metrics to measure the novelty and its impact of scientific publications.

## Usage Guide

### Notebooks
The repository contains scripts and detailed Jupyter notebooks that guide users through each step of the process. The notebooks are particularly beneficial for those aiming to execute specific tasks or a subset of the entire process.

- **`0.tutorial`**: A comprehensive guide that offers a step-by-step walkthrough of all phases, serving as an introductory overview.
- **`1.data-collection`**: Instructions for downloading a custom set of papers from OpenAlex or searching within the Zenodo repository.
- **`2.preprocessing`**: A guide for preprocessing titles and abstracts (and full texts, if available) of a selected set of papers.
- **`3.text-embeddings`** and **`4.cosine-distance`**: Notebooks for generating text embeddings and calculating semantic distance using SPECTER.
- **`5.new-word`**, **`6.new-phrase`**, **`7.new-word-comb`**, **`8.new-phrase-comb`**: Detailed guides for identifying new words, new phrases, new word combinations and new phrase combinations in processed papers.

### Improvements
This repository is optmized to be run on a Server with high computational capabilities (about 500GB of RAM). Cloud services can be used to optimize the whole process. BigQuery is the most valid alternative. ChromaDB is also a valid alternative to store and query embedding data.

### Custom Analysis
Users are encouraged to adapt the code for their specific research needs, ensuring a flexible and customizable approach to analyzing scientific novelty and impact.

## Contribution & Feedback
Contributions to enhance the code and extend its functionalities are warmly welcomed. For any inquiries, issues, or feedback, feel free to open an issue or contact us directly at nicola.melluso@kuleuven.be.
Part of this code is inspired from `https://github.com/sam-arts/respol_patents_code`

## Respect Copyrights
Users are reminded to adhere to copyright regulations and ethical guidelines when utilizing and adapting the provided resources and data.
