# concreteness-prediction

This project investigates computational methods for estimating how concrete or abstract words are (for example, "apple" vs. "justice") in both Modern Standard Arabic and English. It introduces a novel dataset of 202 Arabic nouns with concreteness ratings and aligns these with established English norms. The main approach uses K-Nearest Neighbors (KNN) regression with word embeddings, and compares the results to predictions from ChatGPT. The models are evaluated on held-out test sets, supporting applications in psycholinguistics and natural language processing.

## Repository Structure

- **concreteness_predictor.py, data_processing.py, embedding_loader.py, evaluation.py, main.py**  
  Core modules for data handling, embedding loading, KNN regression, evaluation, and experiment running.

- **analysis/**  
  Folder for thesis figures, qualitative analysis CSVs, and related files.  
  *(No code in this folder is essential for running main experiments.)*

- **ChatGPT/**  
  LLM prediction scripts and results.  
  - ar/, en/: Input word lists and output CSVs for Arabic and English.
  - arabic_concreteness_o4mini.py, english_concreteness_o4mini.py: Scripts for batch prediction with ChatGPT.
  - arabic_decimal_concreteness_o4mini.py: Script for decimal-scale Arabic ratings.
  - Output CSVs: Model predictions per word.

- **datasets/**  
  Data files (full, train, and test splits) for both Arabic and English (ar/, en/).

- **embeddings/**  
  Place downloaded word embeddings here under ar/ and en/.  
  *(Embeddings are not included due to size; see README for download instructions.)*

- **results/**  
  All experiment outputs, organized by language (ar/, en/), including CSVs for test results, cross-validation, etc.

- **subsets/**  
  Data subsets for the subset experiment (eight combinations by concreteness/frequency), plus generate_subsets.py to generate them from the training set.

- **X2Static/**  
  Forked from [epfml/X2Static](https://github.com/epfml/X2Static) for generating static embeddings from contextual models.  
  Contains  additionally ar_wiki_dataset_cleaning.py for creating 2  para and sent corpora  compatible with the X2Static embeddings code.

- **LICENSE, README.md, requirements.txt**  
  License, documentation, and dependency specification.

## Requirements and Installation

All code is written in Python 3.11.

Install dependencies using pip and the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

## Data and Embeddings

### Datasets

* **English:**
  This project uses a filtered subset of 5,448 English nouns from the concreteness norms of [Brysbaert et al. (2014)](https://link.springer.com/article/10.3758/s13428-013-0403-5), following the criteria of [Schulte im Walde and Frassinelli (2022)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2021.796756/full). All splits (full, train, test) are provided in datasets/en/.

* **Arabic:**
  A new dataset of 202 Modern Standard Arabic nouns, each manually annotated with concreteness ratings on a five-point scale. All splits (full, train, test) are provided in datasets/ar/.

* **Subset experiments:**
  The subsets/ folder contains CSVs for predefined groups (extremes, mid-range, high/low frequency, intersections) and a script to generate them from the training set.

---

### Embeddings

**Note:** Embedding files are not included in this repository due to size and licensing.
Please download the required embeddings and place them in the correct subfolders (embeddings/en/ or embeddings/ar/).

#### English Embeddings

* Download pre-trained English **fastText embeddings** from fastText official site [Common Crawl](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip), [Common Crawl subword](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip), [Wikipedia](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip), and [Wikipedia subword](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip).
* **Static transformer-based embeddings** for English can be generated or downloaded as described in the X2Static/ folder (see [epfml/X2Static](https://github.com/epfml/X2Static)).

#### Arabic Embeddings

* Download pre-trained Arabic **fastText embeddings** from fastText official site [Common Crawl](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ar.300.bin.gz), and [Wikipedia](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ar.zip).
* **Static transformer-based embeddings** for Arabic are available upon request (contact the repository owner).
  Alternatively, you can create your own:

  1. Download the latest [Arabic Wikipedia dump](https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles.xml.bz2).
  2. Process with [WikiExtractor](https://attardi.github.io/wikiextractor/).
  3. Use it as described in X2Static/.

---

## Usage

The main experiments and evaluations are run using main.py. All outputs are saved as CSV files in results/{en, ar}/.

### Running Experiments

#### 1. Final Test Set Evaluation

**Purpose:**
Train a KNN regression model on the full training data and evaluate its performance on the held-out test set.
Use this to obtain final performance metrics for a chosen configuration.

**Command:**

```bash
python main.py --lang en --exp final_test
```

(Replace en with ar for Arabic.)

**Output:**
A CSV file containing the main evaluation metrics (RMSE, Spearman, Pearson  correlations).

---

#### 2. KNN Hyperparameter Sweep

**Purpose:**
Evaluate model sensitivity to the number of neighbors k in KNN regression.
Use this to identify the optimal k for your data.

**Command:**

```bash
python main.py --lang en --exp k
```

**Output:**
A CSV with model performance for each k value (5 to 100, step 5).

---

#### 3. Weighting Scheme Comparison

**Purpose:**
Compare uniform vs distance weighting in KNN neighbor averaging.
Assesses whether closer neighbors should be weighted more heavily.

**Command:**

```bash
python main.py --lang en --exp weighting
```

**Output:**
A CSV with results for both weighting schemes.

---

#### 4. Embedding Type Comparison

**Purpose:**
Compare different word embedding sources (fastText, transformer-based/X2Static) for their effect on model performance.
Use this to determine which embedding is best for your language.

**Command:**

```bash
python main.py --lang en --exp embeddings
```

**Output:**
A CSV file with results for each tested embedding.

**To run with a specific embedding file:**

```bash
python main.py --lang en  --model_path embeddings/en/bert_24layer_sent.vec --file_format vec
```

---

#### 5. Subset Training Experiment

**Purpose:**
Train the model on selected subsets of the data (e.g., only highly concrete and highly abstract words, only high-frequency words) and evaluate on the standard test set.

Use this to analyze how training data selection affects generalization.

**Command:**

```bash
python main.py --lang en --exp subsets
```

**Output:**
A CSV with results for each subset (defined in subsets/).

---

#### 6. Per-Word Cross-Validation

**Purpose:**
Perform k-fold cross-validation on the training data, saving a predicted concreteness score for each word in every fold.
Provides word-level out-of-fold predictions for error analysis or visualization.

**Command:**

```bash
python main.py --lang en --exp per_word_cv
```

**Output:**
A CSV with each word’s true and predicted score, and its fold assignment.

### Command-Line Arguments

To view all available options and defaults, run:

```bash
python main.py --help
```

Key arguments include:

* **Required** --lang (en or ar)
* \--exp (see above)
* \--model\_path (path to embedding file)
* \--file\_format (bin or vec)
* \--k, --weighting, --test\_size, etc.

All results and intermediate outputs are written to the results/ directory under the corresponding language subfolder.

---

### ChatGPT-Based Predictions

**Purpose:**
Estimate concreteness scores for a list of words using ChatGPT o4-mini, for either Arabic or English.

**Commands:**

* **English:**

```bash
python ChatGPT/en/english_concreteness_o4mini.py ChatGPT/en/en_words.txt
```

* **Arabic:**

```bash
python ChatGPT/ar/arabic_concreteness_o4mini.py ChatGPT/ar/ar_words.txt
```

* **For decimal-scale Arabic predictions:**

```bash
python ChatGPT/ar/arabic_decimal_concreteness_o4mini.py ChatGPT/ar/ar_words.txt
```

**Output:**
CSV files with one or more model predictions per word in ChatGPT/{ar, en}/.

---
