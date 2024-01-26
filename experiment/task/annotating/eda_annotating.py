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

def eda_annotating(args: argparse.Namespace) -> None:
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
    for idx in tqdm(range(len(train_data_dict['image_names'])), desc='Annotating with EDA...'):
        # Get image_name, caption
        image_name = train_data_dict['image_names'][idx]
        gold_caption = train_data_dict['captions'][idx]

        # Apply EDA
        eda_sentences = run_eda(gold_caption)
        result_sentences = [gold_caption] + eda_sentences

        for i in range(len(result_sentences)):
            # Tokenize
            tokenized = en_tokenizer(result_sentences[i], padding='max_length', truncation=True,
                                     max_length=args.max_seq_len, return_tensors='pt')

            # Append to data_dict
            save_data['image_names'].append(image_name)
            save_data['captions'].append(result_sentences[i])
            save_data['caption_numbers'].append(i+1) # 1 is gold caption
            save_data['input_ids'].append(tokenized['input_ids'].squeeze())

    save_name = 'train_EDA_EN.pkl'
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

def random_deletion(words: list, p: float) -> list:
    """
    Randomly delete words from the sentence with probability p.
    Args:
        words (list): The list of words in the sentence.
        p (float): The probability of deleting a word.
    Returns:
        list: The list of words in the sentence after deletion.
    """

    # Obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # Randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1) # Generate a random number between 0 and 1
        if r > p: # If the random number is greater than p, then keep the word
            new_words.append(word)
        else:
            #print("deleted", word) # If the random number is less than p, then delete the word
            continue

    # If you end up deleting all words, just return a random word from the original sentence
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

def random_swap(words: list , n: int) -> list:
    """
    Randomly swap two words in the sentence n times.
    Args:
        words (list): The list of words in the sentence.
        n (int): The number of times to swap two words.
    Returns:
        list: The list of words in the sentence after swapping.
    """

    new_words = words.copy()

    for _ in range(n): # Swap the words n times
        new_words = swap_word(new_words)

    return new_words

def swap_word(new_words: list) -> list:
    """
    This is a sub-function of random swap to swap two words in the sentence.
    Args:
        new_words (list): The list of words in the sentence.
    Returns:
        list: The list of words in the sentence after swapping.
    """

    random_idx_1 = random.randint(0, len(new_words)-1) # Get a random index
    random_idx_2 = random_idx_1 # Get another random index
    counter = 0

    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1) # Make sure the two random indices are different
        counter += 1 # If the two random indices are the same, then try again
        if counter > 3: # If you try more than 3 times, then just return the original sentence
            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] # Swap the words

    return new_words

def random_insertion(words: list, n: int) -> list:
    """
    Randomly insert n words into the sentence.
    Args:
        words (list): The list of words in the sentence.
        n (int): The number of words to be inserted.
    Returns:
        list: The list of words in the sentence after insertion.
    """

    new_words = words.copy()

    for _ in range(n):
        add_word(new_words)

    return new_words

def add_word(new_words: list) -> None:
    """
    This is a sub-function of random insertion to insert a word into the sentence.
    Args:
        new_words (list): The list of words in the sentence.
    """

    synonyms = []
    counter = 0

    while len(synonyms) < 1: # If there are no synonyms, then try again with a different word
        random_word = new_words[random.randint(0, len(new_words)-1)] # Get a random word from the sentence
        synonyms = get_synonyms(random_word) # Get the synonyms of the random word

        counter += 1
        if counter >= 10:
            return

    random_synonym = synonyms[0] # Pick a random synonym from the list of synonyms

    random_idx = random.randint(0, len(new_words)-1) # Get a random index
    new_words.insert(random_idx, random_synonym) # Insert random synonym of a word at the random index

def run_eda(sentence: str) -> list: # list[str]
    """
    Main function to perform EDA.
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

    # Apply each of the EDA operations to get 4 augmented sentences
    # Synonym replacement
    n_sr = max(1, int(0.15 * len_words))
    new_words = synonym_replacement(words, n_sr)
    augmented_sentence.append(' '.join(new_words))

    # Random swap
    n_rs = max(1, int(0.15 * len_words))
    new_words = random_swap(words, n_rs)
    augmented_sentence.append(' '.join(new_words))

    # Random insertion
    n_ri = max(1, int(0.15 * len_words))
    new_words = random_insertion(words, n_ri)
    augmented_sentence.append(' '.join(new_words))

    # Random deletion
    n_rd = max(1, int(0.15 * len_words))
    new_words = random_deletion(words, n_rd)
    augmented_sentence.append(' '.join(new_words))

    assert len(augmented_sentence) == 4 # Make sure we have 4 augmented sentences
    return augmented_sentence
