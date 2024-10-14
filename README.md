# Optimizing Language Models through Enhanced Fine-Tuning with Data Augmentation Techniques

Paper: [Optimizing Language Models through Enhanced Fine-Tuning with Data Augmentation Techniques.pdf](https://github.com/DanielDaCosta/FineTunedBERT-Data-Augmentation/blob/main/Optimizing%20Language%20Models%20through%20Enhanced%20Fine-Tuning%20with%20Data%20Augmentation%20Techniques.pdf)

# Abstract
Text classification, one of the core tasks of Natural Language Processing (NLP), encounters challenges when evaluating models in out-of-distribution (OOD) contexts. Addressing these challenges requires the application of specialized techniques to enhance model performance. This paper analyzes the efficacy of a fine-tuned iteration of BERT on a custom OOD dataset, utilizing data augmentation techniques to bolster its performance and showcasing the efficacy of this technique. Through a comparative analysis with DistilBERT and GPT-3.5, the paper demonstrates that comparable results can be achieved with a 40\% smaller model, emphasizing the potential for efficiency gains without sacrificing performance.

# Introduction
Fine-tuning a model a pre-trained model on a downstream task is a common procedure in the NLP space, as it facilitates achieving higher performance with minimal effort. However, one important aspect to consider is that, in real-world scenarios, test data often deviates from the training data distribution. As a result, ensuring that the model exhibits robust performance on datasets with both similar and divergent distributions is crucial.

In this paper, we go over fine-tuning a BERT model on binary classification tasks, testing its performance on a specifically crafted out-of-distribution dataset and discussing the reasons behind the observed decline in the model's effectiveness under these circumstances. Furthermore, the paper encompasses the application of a data augmentation technique involving expanding the training set with out-of-distribution data, followed by a subsequent round of fine-tuning.

We further extend our investigation by applying the previously outlined procedure to DistilBERT, a model that is 40% smaller, highlighting the trade-off between efficiency and performance.  To validate the model accuracy, we use GPT-3.5 as a baseline in a zero-shot setting on a small subset of the dataset to verify the model's performance.

The results showcase an enhancement in performance on the out-of-distribution (OOD) dataset after the integration of data augmentation. However, this improvement is accompanied by a comparatively modest decrease in performance on the original dataset.  Moreover, the study emphasizes that employing DestilBERT, a smaller model that can be trained 50% faster, enables the preservation of the model's performance in a similar setting.

# Getting Started

## Dataset
IMDB Dataset: Large Movie Review Dataset. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well.

https://huggingface.co/datasets/stanfordnlp/imdb

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
Files within `./CARC_output` folder

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
# Evaluation of the original DistilBERT model on transformed data
python3 main_distilBERT.py --eval_augmented --model_dir ./out_distilbert

# Evaluation augmented DistilBERT model on original data
python3 main_distilBERT.py --eval_augmented --model_dir ./out_distilbert_augmented
```
