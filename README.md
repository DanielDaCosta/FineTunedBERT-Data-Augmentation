# Homework 3

**Name:** Daniel Pereira da Costa

**USC ID:** 3777473693

# Getting Started

## Installation
Python 3.11.5:
- torch==2.1.0
- datasets==2.14.6
- tqdm==4.66.1
- transformers==4.35.0
- evaluate==0.4.1
- gensim==4.3.2
- nltk==3.8.1

Or you can install them by running:

```
pip install -r requirements.txt
```

## Files
- `main.py`: script for fine-tuning and evaluation BERT on the original or transformed dataset.
- `main_distilBERT.py`: script for fine-tuning and evaluation DistilBERT on the original or transformed dataset.
- `utils.py`: support script that has all of the transformations to created the out-of-distributions dataset
- `word2vec_model.bin`: word2vec embeddings used for synonym replacement
- `main_GPT.ipynb`: Jupyter Notebook for running GPT-3.5 evalaluations of Original (Sample) and Transformed (Sample) datasets as well as BERT and DistilBERt

**Predicton files**
Files within `./pred` folder

BERT:
- `out_original.txt`: Fine-tuned BERT on original dataset
- `out_original_transformed.txt`: Fine-tuned BERT on transformed dataset
- `out_augmented_original.txt`: Fine-tuned augmented BERT on original dataset
- `out_augmented_transformed`: Fine-tuned augmented BERT on transformed dataset
- `out_100_original.txt`:  Fine-tuned BERT predictions on the first 100 rows of the original dataset
- `out_augmented_100_transformed.txt`:  Fine-tuned augmented BERT predictions on the first 100 rows of the transformed dataset

DistilBERT:
- `out_distilbert_original.txt`: Fine-tuned DistilBERT on original dataset
- `out_distilbert_original_transformed.txt`: Fine-tuned DistilBERT on transformed dataset
- `out_distilbert_augmented_original.txt`: Fine-tuned augmented DistilBERT on original dataset
- `out_distilbert_augmented_transformed.txt`: Fine-tuned augmented DistilBERT on transformed dataset
- `out_distilbert_100_original.txt`:  Fine-tuned DistilBERT predictions on the first 100 rows of the original dataset
- `out_distilbert_augmented_100_transformed.txt`:  Fine-tuned augmented DistilBERT predictions on the first 100 rows of the transformed dataset

GPT3.5 (zero-shot):
- `gpt_out_original.txt`: prediction on the first 100 rows of the original dataset
- `gpt_out_transformed.txt`: prediction on the first 100 rows of the transformed dataset


**CARC Output Files**
`./CARC_output/`: contain all of CARC outputs for each training and evaluation that were executed

# Usage

## Fine-Tuning and Evaluating on Original Dataset
```python
python3 main.py --train --eval
```
Outputs: 
- out/:  model tensors
- out_original.txt: predictions

```python
python3 main_distilBERT.py --train --eval
```
Outputs: 
- out_distilbert/:  model tensors
- out_distilbert_original.txt: predictions

## Fine-Tuning and Evaluating on Transformed Dataset
```python
python3 main.py --train_augmented --eval_augmented
```
Outputs: 
- out_augmented/:  model tensors
- out_augmented_original.txt: predictions


```python
python3 main_distilBERT.py --train_augmented --eval_augmented
```
Outputs: 
- out_distilbert_augmented/:  model tensors
- out_distilbert_augmented_original.txt: predictions

## Evaluations

```python
# Evaluation original BERT model on transformed data
python3 main.py --eval_augmented --model_dir ./out

# Evaluation augmented BERT model on original data
python3 main.py --eval_augmented --model_dir ./out_augmented
```


```python
# Evaluation original DistilBERT model on transformed data
python3 main_distilBERT.py --eval_augmented --model_dir ./out_distilbert

# Evaluation augmented DistilBERT model on original data
python3 main_distilBERT.py --eval_augmented --model_dir ./out_distilbert_augmented
```