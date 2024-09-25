import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import string
from nltk import pos_tag
import re
from gensim.models import Word2Vec
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

random.seed(0)


def example_transform(example):
    
    example["text"] = example["text"].lower()
    return example


##################
# Sythetic Typos #
##################

keyboard_layout = {
    'Q': ['W', 'A', 'S'], 'W': ['Q', 'E', 'A', 'S', 'D'], 'E': ['W', 'R', 'S', 'D', 'F'], 'R': ['E', 'T', 'D', 'F', 'G'], 'T': ['R', 'Y', 'F', 'G', 'H'], 
    'Y': ['T', 'U', 'G', 'H', 'J'], 'U': ['Y', 'I', 'H', 'J', 'K'], 'I': ['U', 'O', 'J', 'K', 'L'], 'O': ['I', 'P', 'K', 'L'], 'P': ['O', 'L'],
    'A': ['Q', 'W', 'S', 'Z', 'X'], 'S': ['Q', 'W', 'E', 'A', 'D', 'Z', 'X', 'C'], 'D': ['W', 'E', 'R', 'S', 'F', 'X', 'C', 'V'], 
    'F': ['E', 'R', 'T', 'D', 'G', 'C', 'V', 'B'], 'G': ['R', 'T', 'Y', 'F', 'H', 'V', 'B', 'N'], 'H': ['T', 'Y', 'U', 'G', 'J', 'B', 'N', 'M'], 
    'J': ['Y', 'U', 'I', 'H', 'K', 'N', 'M'], 'K': ['U', 'I', 'O', 'J', 'L', 'M'], 'L': ['I', 'O', 'P', 'K'], 'Z': ['A', 'S', 'X'], 'X': ['Z', 'A', 'S', 'D', 'C'],
    'C': ['X', 'S', 'D', 'F', 'V'], 'V': ['C', 'D', 'F', 'G', 'B'], 'B': ['V', 'F', 'G', 'H', 'N'], 'N': ['B', 'G', 'H', 'J', 'M'], 'M': ['N', 'H', 'J', 'K'],
    'I': ['U', 'O', 'J', 'K', 'L'], 'W': ['Q', 'E', 'A', 'S', 'D']
}

def typo_qwert(word: str, index: int):
    """Replace words close to each other in the QWERT keyboard
    :param word: 
    :param index: index of caracter to be replaced
    """
    # 
    char = word[index]
    if char.upper() in keyboard_layout.keys():
        random_typo = random.choice(keyboard_layout[char.upper()]).lower()
        word = word[:index] + random_typo + word[index+1:]

    return word

def typo_swap_characeters(word: str, index: int):
    """Swap character at index with index + 1
    :param word: 
    :param index: index of caracter to be replaced
    """
    char_list = list(word)
    next_index = index + 1
    char_list[index], char_list[next_index] = char_list[next_index], char_list[index]
    word_swapped = "".join(char_list)
    return word_swapped

def typo_remove_character(word: str, index: int):
    """Delete a character
    :param word: 
    :param index: index of caracter to be replaced
    """
    word = word[:index] + word[index + 1:]
    return word

def typo_add_character(word: str, index: int):
    """Add character 
    :param word: 
    :param index: index of caracter to be replaced
    """
    random_char = random.choice(string.ascii_letters)

    return word[:index] + random_char + word[index:]

def add_typos(text: str, pct_typos: float = 0.4, typos_list=['swap', 'remove', 'add', 'qwert']) -> str:
    """Replace {pct_typos}% of words in a sentence with their syntetic generated typo
    :param text (str):
    :param pct_typos (float): percentage of words to be modified. Between 0 and 1
    :return: transformed text
    """
    # Tokenize sentence
    word_list  = word_tokenize(text)


    # word_list words that have at least 3 characters
    word_list_indices_filtered = [i for i in range(len(word_list)) if len(word_list[i]) >= 3]

    n_tokens = len(word_list_indices_filtered) # only count the tokens that have >= characters
    n_typos = int(n_tokens*pct_typos) # number of typos

    # Typos functions
    typos_functions = {
        'swap': typo_swap_characeters,
        'remove': typo_remove_character,
        'add': typo_add_character,
        'qwert': typo_qwert
    }

    # Select indices based on pct_typos
    word_indices = [index for index in range(len(word_list)) if index in word_list_indices_filtered]
    selected_indices = random.sample(word_indices, k = n_typos)

    # Transformed words
    transformed_words = {}
    for index in selected_indices:

        word = word_list[index]
        typo_index= random.randint(0, len(word) - 2)
        # Randomly select typo technique
        selected_typo = random.choice(typos_list)

        # Apply typo function
        word_transformed = typos_functions[selected_typo](word, typo_index)
        transformed_words[index] = word_transformed

    
    # Replace orginal words with sythetic generated typos
    for i, new_word in transformed_words.items():
        word_list[i] = new_word

    # Put sentence together again:
    text = TreebankWordDetokenizer().detokenize(word_list)
    
    
    return text


############
# Synonyms #
############
def get_wordnet_pos(treebank_tag): # Convert POS-Tag from treebank_tag to Wordnet tags
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None  # If no match is found, return None
    

def replace_with_synonym(text: str, pct_synonyms: float = 0.4) -> str:
    """Replace {pct_synonyms}% of words in a sentence with their synonyms
    :param text (str):
    :param pct_synonyms (float): percentage of words to be replaced. Between 0 and 1
    :return: transformed text
    """
    # Tokenize sentence
    word_list  = word_tokenize(text)
    n_tokens = len(word_list)
    n_synonyms = int(n_tokens*pct_synonyms)

    # Perform POS tagging to help selecting synonym
    pos_tags = pos_tag(word_list)
    syn_dict = {}
    for i in range(n_synonyms):
        chosen_word = random.choice(word_list)
        synonyms = wn.synsets(chosen_word)

        # If chosen word doesn't have a synonym
        # retry until finding a word that does.
        max_retry = len(word_list)
        retry = 0 
        while (len(synonyms) == 0 and retry <= max_retry):
            chosen_word = random.choice(word_list)
            synonyms = wn.synsets(chosen_word)

            if len(synonyms) > 0:
                # Print the word and its POS tag
                chosen_word_tag = [tag for word, tag in pos_tags if word == chosen_word][0]
                chosen_word_wordnet_tag = get_wordnet_pos(chosen_word_tag)
                
                selected_synonym = chosen_word
                for syn in synonyms:
                    # Only consider synonyms that have the same POS_TAG
                    if syn.pos() == chosen_word_wordnet_tag:
                        for lemma in syn.lemmas():
                            if (lemma.name() != chosen_word):
                                # Select the first synonym that appears
                                selected_synonym = lemma.name().replace('_', ' ') # Replace underscores for multi-word synonyms
                                break
                        break
                    break
                if selected_synonym == chosen_word:
                    synonyms = [] # continue while loop
                else:
                    # print("selected_synonym: ", selected_synonym)
                    syn_dict[chosen_word] = selected_synonym

    # return syn_dict
    # Replace selected words
    for old, new in syn_dict.items():
        text = text.replace(old, new)
    return text


########################
# Expand Constractions #
########################
def expand_contradictions(text):

    contraction_mapping = {
        "won't": "will not",
        "can't": "can not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am",
        "won\'t": "will not",
        "can\'t": "can not",
        "n\'t": " not",
        "isn\'t": "is not",
        "\'re": " are",
        "\'s": " is",
        "\'d": " would",
        "\'ll": " will",
        "\'ve": " have",
        "\'m": " am"
    }

    pattern = re.compile(r"\b(?:" + "|".join(re.escape(contraction) for contraction in contraction_mapping.keys()) + r")\b")
    text = pattern.sub(lambda x: contraction_mapping[x.group()], text)
    
    return text

################
# Random Swaps #
################

def random_swap(text: str, pct_swap: float = 0.2) -> str:
    """Swap {pct_swap}% of words in a sentence with a random chosen word
    :param text (str):
    :param pct_swap (float): percentage of words to be swapped. Between 0 and 1
    :return: transformed text
    """
    # Tokenize sentence
    word_list  = word_tokenize(text)

    n_tokens = len(word_list)
    n_swaps = int(n_tokens*pct_swap)
    for i in range(n_swaps):
        # Get indices
        index_1, index_2 = random.sample(range(len(word_list)), 2)
        # Swap words
        word_list[index_1], word_list[index_2] = word_list[index_2], word_list[index_1]


    # Put sentence together again:
    text = TreebankWordDetokenizer().detokenize(word_list)
    return text

###################
# Random Deletion #
###################

def random_deletion(text: str, pct_deletion: float = 0.1) -> str:
    """Remove {pct_deletion}% of words in a sentence
    :param text (str):
    :param pct_deletion (float): percentage of words to be deleted. Between 0 and 1
    :return: transformed text
    """
    # Tokenize sentence
    word_list  = word_tokenize(text)

    n_tokens = len(word_list)
    n_deletion = int(n_tokens*pct_deletion)

    indexes_deleted = random.sample(range(len(word_list)), n_deletion)
    for index in sorted(indexes_deleted, reverse=True):
        word_list.pop(index)


    # Put sentence together again:
    text = TreebankWordDetokenizer().detokenize(word_list)
    return text


def replace_with_synonym_word2vec(text: str, pct_synonyms: float = 0.4, word2vec_file: str = "word2vec_model.bin") -> str:
    """Replace {pct_synonyms}% of words in a sentence with their synonyms based on Word2Vec cosine similarity
    :param text (str):
    :param pct_synonyms (float): percentage of words to be replaced. Between 0 and 1
    :param word2vec_file (str): word2vec file
    :return: transformed text
    """
    # Tokenize sentence
    word_list  = word_tokenize(text)

    # Read word2vec
    loaded_model = Word2Vec.load("word2vec_model.bin")
    vocabulary = loaded_model.wv.index_to_key
    
    # word_list words that have are in vocabulary
    word_list_indices_filtered = [i for i in range(len(word_list)) if word_list[i] in vocabulary]

    n_tokens = len(word_list_indices_filtered) # only count the tokens that are in vocabulary
    n_synonyms = int(n_tokens*pct_synonyms)


    # Select indices based on pct_typos
    word_indices = [index for index in range(len(word_list)) if index in word_list_indices_filtered]
    selected_indices = random.sample(word_indices, k = n_synonyms)

    for chosen_index in selected_indices:
        chosen_word = word_list[chosen_index]

        synonym = loaded_model.wv.most_similar(chosen_word, topn=1)[0][0]
        word_list[chosen_index] = synonym

    # Put sentence together again:
    text = TreebankWordDetokenizer().detokenize(word_list)
    return text

def custom_transform(example):
    
    ################################
    ##### YOUR CODE BEGINGS HERE ###
    
    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.
    
    # You should update example["text"] using your transformation
    
    # Select transformation
    selected_transformation = random.choice(["typo", "synonym", "swap", "deletion", "synonym_word2vec"])

    text = example["text"]
    if selected_transformation == "typo":
        text = add_typos(text)
    elif selected_transformation == "synonym":
        text = replace_with_synonym(text)
    elif selected_transformation == "swap":
        text = random_swap(text)
    elif selected_transformation == "synonym_word2vec":
        text = replace_with_synonym_word2vec(text)
    else:
        text = random_deletion(text)

    # Apply contractions to all of them
    text = expand_contradictions(text)
    example["text"] = text
    ##### YOUR CODE ENDS HERE ######

    
    return example



