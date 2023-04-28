# Standard Library Modules
import os
import sys
import time
import pickle
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarning for pandas
import re
import random
import argparse
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet
# Huggingface Modules
from transformers import AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path
from task.captioning.preprocessing import load_caption_data

def synonym_annotating(args: argparse.Namespace) -> None:
    # Define tokenizer - we use bart tokenizer because it has start and end token
    en_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

    # Define data_dict
    with open(os.path.join(args.preprocess_path, 'captioning', args.task_dataset, 'train_ORIGINAL_EN.pkl'), 'rb') as f:
        loaded_data = pickle.load(f)

    train_data_dict = {
        'image_names': [],
        'caption_numbers': [],
        'captions': [],
        'all_captions': [],
        'input_ids': [],
        'tokenizer': en_tokenizer,
    }

    # gather only caption_number == 1
    for idx in range(len(loaded_data['caption_numbers'])):
        if loaded_data['caption_numbers'][idx] == 1:
            train_data_dict['image_names'].append(loaded_data['image_names'][idx])
            train_data_dict['caption_numbers'].append(loaded_data['caption_numbers'][idx])
            train_data_dict['captions'].append(loaded_data['captions'][idx])
            train_data_dict['all_captions'].append(loaded_data['all_captions'][idx])
            train_data_dict['input_ids'].append(loaded_data['input_ids'][idx])
        else:
            continue

    save_data = {
        'image_names': [],
        'caption_numbers': [],
        'captions': [],
        'all_captions': [],
        'input_ids': [],
        'tokenizer': en_tokenizer,
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, 'captioning', args.task_dataset)
    check_path(preprocessed_path)
    for idx in tqdm(range(len(train_data_dict['image_names'])), desc='Annotating with SR...'):
        # Get image_name, caption
        image_name = train_data_dict['image_names'][idx]
        gold_caption = train_data_dict['captions'][idx]

        # Apply SR
        synonym_sentences = run_synonym_replacement(gold_caption)
        result_sentences = [gold_caption] + synonym_sentences

        for i in range(len(result_sentences)):
            # Tokenize
            tokenized = en_tokenizer(result_sentences[i], padding='max_length', truncation=True,
                                     max_length=args.max_seq_len, return_tensors='pt')

            # Append to data_dict
            save_data['image_names'].append(image_name)
            save_data['captions'].append(result_sentences[i])
            save_data['caption_numbers'].append(i+1) # 1 is gold caption
            save_data['input_ids'].append(tokenized['input_ids'].squeeze())

    save_name = 'train_SR_EN.pkl'
    with open(os.path.join(preprocessed_path, save_name), 'wb') as f:
        pickle.dump(save_data, f)
        print(f'Saved {save_name} at {preprocessed_path}')
        print(len(save_data['image_names']))

# List of stopwords
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
            'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who',
            'whom', 'this', 'that', 'these', 'those', 'am',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
            'because', 'as', 'until', 'while', 'of', 'at',
            'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don',
            'should', 'now', '']

def synonym_replacement(words: list, n: int) -> list:
    """
    Replace n words in the sentence with synonyms from wordnet.
    Args:
        words (list): The list of words in the sentence.
        n (int): The number of words to be replaced.
    Returns:
        list: The list of words in the sentence after replacement.
    """

    new_words = words.copy()

    random_word_list = list(set([word for word in words if word not in stop_words])) # Exclude stop words from being replaced
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word) # Get the synonyms of the words which are not stop words
        if len(synonyms) >= 1: # If there are no synonyms, then skip the word
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: # Only replace up to n words
            break

    # This is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_synonyms(word: str) -> list:
    """
    This is a sub-function of synonym replacement to get synonyms of given word.
    Args:
        word (str): The word to be replaced.
    Returns:
        list: The list of synonyms of the given word.
    """
    synonyms = set()

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm']) # Remove special characters
            synonyms.add(synonym)

    if word in synonyms:
        synonyms.remove(word) # Remove the original word from the set of synonyms

    return list(synonyms)

def run_synonym_replacement(sentence: str) -> list: # list[str]
    """
    Main function to perform SR.
    Default value of alpha is 0.15 - Perturb 15% of words in the sentence
    Args:
        sentence (str): The sentence to be augmented.
    Returns:
        str: The augmented sentence.
    """
    words = sentence.split(' ')
    words = [word for word in words if word != '']
    len_words = len(words)
    augmented_sentence = []

    # Synonym replacement
    n_sr = max(1, int(0.15 * len_words))

    for i in range(4):
        n_sr_ = n_sr + random.randint(0, 2) # 15% + 0~2 words
        new_words = synonym_replacement(words, n_sr_)
        augmented_sentence.append(' '.join(new_words))

    assert len(augmented_sentence) == 4 # Make sure we have 4 augmented sentences
    return augmented_sentence
