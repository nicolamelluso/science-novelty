# Measuring Novelty and Impact in Science using Natural Language Processing

If you use the code or data, please cite the following paper: 
 > *Arts S., Melluso N., Veugelers R. (2023). Beyond Citations: Text-Based Metrics for Assessing Novelty and its Impact in Scientific Publications*

## Overview

This repository contains Python scripts and Jupyter notebooks aimed at assessing novelty and its impact in scientific publications. This repository has a twofold objective:
- Reproduce the results from the original paper. In this paper data from Microsoft Academic Graph (MAG) (now OpenAlex) is used for the entire population of papers from 1800 to 2020. Data for this is available in the Zenodo repository: https://zenodo.org/record/8283353
- Reproduce the analysis (preprocessing, metrics etc...) for a custom set of papers. Users may use this code to measure novelty and its impact on a subset of papers crafted for their own research. We encourage to do it.


Here is a graphical description of what this repository does.
![Science Novelty Schema](https://github.com/nicolamelluso/science-novelty/blob/main/ScienceNovelty-schema.png)

The entire process is organized in the following steps:
1. **Data Collection**
2. **Preprocessing**
3. **Text Embeddings**
4. **Cosine Distance**
5. **New Word**
6. **New Bigrams**
7. **New Trigrams**
8. **New Word Combinations**

Each step serves as pivotal to gather the text-based metrics of novelty and its impact from scientific publications.

## How to use it

The repository contains scripts and Jupyter notebooks. Notebooks contain detailed information and explanation on the process.

This repository is mainly useful for users that want to perform their own subset of tasks. Scripts are available for different usages. For example:
- If some wants to download a custom set papers from OpenAlex or search within the Zenodo repository of the corresponding paper of this repository can follow the instructions from the notebook **1.data-collection**
- If some wants to preprocess the title and abstract (and the full text if available) from a custom set of papers following the procedure described in the corresponding paper of this repository can use the notebook **2.preprocessing**
- If some wants to get the embeddings of a custom set papers and then calculate their cosine similarity can follow the notebooks **3.text-embeddings** and **3.cosine-distance**
- Is some wants to identify new words, new bigrams, new trigrams and new word combs from a custom set of processed papers can use the notebooks **5.new-word**, **6.new-bigram**, **7.new-trigram** and **5.new-word-comb**.

To this end the notebooks are organized as follow:

- A general Jupyter notebook called **`0.tutorial`** in which is shown how to perform all the phases step-by-step. The **`0.tutorial`** notebook is a general overview of the entire process.
- Each phase is deployed and described in details in the other remaining notebooks


# Contribution & Feedback
We welcome contributions to improve the code and its functionality. If you have any issues or feature requests, or if you'd like to provide feedback, please feel free to reach out, open issues. Email us at nicola.melluso@kuleuven.be.

Respect copyrights.
