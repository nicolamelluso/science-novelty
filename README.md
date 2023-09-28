# Measuring Novelty and its Impact in Science using Natural Language Processing

## Citation

If you use the code from this repository, please cite the following paper: 
 > *Arts S., Melluso N., Veugelers R. (2023). Beyond Citations: Text-Based Metrics for Assessing Novelty and its Impact in Scientific Publications*

## Overview

This repository is dedicated to the assessement of novelty and its impact of scientific publications, employing Python scripts and Jupyter notebooks. It is designed with a dual purpose:

- **Reproduce Results**: Replicate the findings of the original paper, which analyzes data from the Microsoft Academic Graph (MAG), now OpenAlex, encompassing a comprehensive collection of papers from 1800 to 2020. The data can be accessed here: https://zenodo.org/record/8283353.
- **Custom Analysis**: Enable users to apply the analysis, including preprocessing and metrics calculation, to a tailored set of papers for individual research needs.

![Science Novelty Schema](https://github.com/nicolamelluso/science-novelty/blob/main/data/ScienceNovelty-schema.png)

The methodology is systematically organized into the following segments:
1. **Data Collection**
2. **Preprocessing**
3. **Text Embeddings**
4. **Cosine Distance**
5. **New Word**
6. **New Bigrams**
7. **New Trigrams**
8. **New Word Combinations**

Each segment is integral for extracting text-based metrics to measure the novelty and its impact of scientific publications.

## Usage Guide

### Notebooks
The repository contains scripts and detailed Jupyter notebooks that guide users through each step of the process. The notebooks are particularly beneficial for those aiming to execute specific tasks or a subset of the entire process.

- `0.tutorial`**: A comprehensive guide that offers a step-by-step walkthrough of all phases, serving as an introductory overview.
- **`data-collection`**: Instructions for downloading a custom set of papers from OpenAlex or searching within the Zenodo repository.
- **`preprocessing`**: A guide for preprocessing titles and abstracts (and full texts, if available) of a selected set of papers.
- **`text-embeddings`** and 4.cosine-distance`**: Notebooks for generating text embeddings and calculating cosine similarity.
- **`new-word`**, **`6.new-bigram`**, **`7.new-trigram`**, **`8.new-word-comb`**: Detailed guides for identifying new lexical elements and combinations in processed papers.

This repository is mainly useful for users that want to perform their own subset of tasks. Scripts are available for different usages. For example:
- If some wants to download a custom set papers from OpenAlex or search within the Zenodo repository of the corresponding paper of this repository can follow the instructions from the notebook **`1.data-collection`**
- If some wants to preprocess the title and abstract (and the full text if available) from a custom set of papers following the procedure described in the corresponding paper of this repository can use the notebook **`2.preprocessing`**
- If some wants to get the embeddings of a custom set papers and then calculate their cosine similarity can follow the notebooks **`3.text-embeddings`** and **`3.cosine-distance`**
- Is some wants to identify new words, new bigrams, new trigrams and new word combs from a custom set of processed papers can use the notebooks **`5.new-word`**, **`6.new-bigram`**, **`7.new-trigram`** and **`5.new-word-comb`**.

### Custom Analysis
Users are encouraged to adapt the code for their specific research needs, ensuring a flexible and customizable approach to analyzing scientific novelty and impact.
To this end the notebooks are organized as follow:

## Contribution & Feedback
Contributions to enhance the code and extend its functionalities are warmly welcomed. For any inquiries, issues, or feedback, feel free to open an issue or contact us directly at nicola.melluso@kuleuven.be.
Part of this code is inspired from `https://github.com/sam-arts/respol_patents_code`

## Respect Copyrights
Users are reminded to adhere to copyright regulations and ethical guidelines when utilizing and adapting the provided resources and data.
