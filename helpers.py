#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File Name: helpers.py
Author: Alexandre Donciu-Julin
Date: 2024-10-02
Description: Helper methods to interact with the dataset quickly.
"""

# Import statements
import os
import re
import pandas as pd
from random import randint
import pickle
from unidecode import unidecode
from tqdm import tqdm
tqdm.pandas()

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

import spacy
nlp = spacy.load("en_core_web_sm")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

SEPARATOR = 100 * '-'


def save_to_pickle(filename: str, data: pd.Series) -> None:
    """Pickle a dataset to disk for future use

    Args:
        filename (str): file path to save the data
        data (pd.Series): dataset to save
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_from_pickle(filename: str) -> pd.Series:
    """Loads a pickled dataset from disk to avoid re-importing it

    Args:
        filename (str): file path to load the data from

    Returns:
        pd.Series: dataset loaded from the pickle file
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def pickled_data_exists(filenames: list) -> bool:
    """Checks if all the files in the list exist

    Args:
        filenames (list): list of file paths to check

    Returns:
        bool: True if all files exist, False otherwise
    """
    return all(os.path.exists(filename) for filename in filenames)


def print_text(X: pd.Series, y: pd.Series, idx: int = None) -> None:
    """Print the text and label of an email at the given index (or random)

    Args:
        X (pd.Series): features
        y (pd.Series): labels
        idx (int, optional): index. Defaults to None (random).
    """

    try:
        print(SEPARATOR)
        if idx is None:
            idx = randint(0, X.shape[0])
        print(f"[{idx}]", X[idx], "-->", y[idx])
        print(SEPARATOR)

    except Exception as e:
        print(f"Can't print email contents: {e}")


def clean_text(text: str) -> str:
    """Clean text by removing special characters, accents, numbers, and converting to lowercase

    Args:
        text (str): text to clean

    Returns:
        str: cleaned text
    """

    # Remove special characters
    text = re.sub(r'[^A-Za-z\s]', '', text)

    # Normalize accents
    text = unidecode(text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Substitute multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    # Convert to lowercase
    text = text.lower().strip()

    return text


def remove_stopwords(text: str) -> str:
    """Remove stopwords from the text

    Args:
        text (str): text to remove stopwords from

    Returns:
        str: text without stopwords
    """

    # get english stopwords
    stopwords_sp = stopwords.words("english")

    # tokenize the text by splitting on spaces
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords_sp]

    return ' '.join(tokens)


def remove_short_sentences(X: pd.Series, y:pd.Series, min_len: int = 3) -> tuple[pd.Series, pd.Series]:
    """Remove short sentences from the dataset

    Args:
        X (pd.Series): features
        y (pd.Series): labels
        min_len (int, optional): Minimum length of a sentence in words. Defaults to 3.

    Returns:
        tuple[pd.Series, pd.Series]: filtered features and labels
    """

    def filter_short_texts(text):
        return len(text.split()) >= min_len

    X_filtered = X[X.progress_apply(filter_short_texts)].reset_index(drop=True)
    y_filtered = y[X.progress_apply(filter_short_texts)].reset_index(drop=True)

    return X_filtered, y_filtered


def lemmatize_text(text: str) -> str:
    """Lemmatize the text

    Args:
        text (str): text to lemmatize

    Returns:
        str: lemmatized text
    """
    doc = nlp(text, disable=['ner', 'parser'])

    # Extract the lemmatized version of each token
    return " ".join([token.lemma_ for token in doc])


def clean_dataset(X: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Clean the dataset by applying the following steps:
    - Cleaning: remove special characters, accents, numbers, and convert to lowercase
    - Removing stopwords: remove stopwords from the text
    - Removing short sentences: remove sentences with less than 3 words
    - Lemmatization: lemmatize the text

    Args:
        X (pd.Series): features
        y (pd.Series): labels

    Returns:
        tuple[pd.Series, pd.Series]: cleaned features and labels
    """

    # apply cleaning
    print("Cleaning Text")
    X = X.progress_apply(clean_text)

    # apply removing stopwords
    print("Removing Stopwords")
    X = X.progress_apply(remove_stopwords)

    # apply remove short sentences (less than 3 words)
    print("Removing short sentences")
    X, y = remove_short_sentences(X, y)

    # apply lemmatization
    print("Lemmatizing Text")
    X = X.progress_apply(lemmatize_text)

    return X, y


def vectorize_bow(X_train: pd.Series, X_test: pd.Series = None) -> tuple[pd.Series, pd.Series] | pd.Series:
    """Vectorize the text using Bag of Words (CountVectorizer)

    Args:
        X_train (pd.Series): training dataset
        X_test (pd.Series, optional): testing dataset. Defaults to None.

    Returns:
        tuple[pd.Series, pd.Series] | pd.Series: vectorized datasets
    """

    # Initialize CountVectorizer
    vectorizer = CountVectorizer()

    # fit and transform messages
    X_train_vect = vectorizer.fit_transform(X_train)

    if X_test is not None:
        X_test_vect = vectorizer.transform(X_test)
        return X_train_vect, X_test_vect

    return X_train_vect


def vectorize_tfidf(X_train: pd.Series, X_test: pd.Series = None) -> tuple[pd.Series, pd.Series] | pd.Series:
    """Vectorize the text using TF-IDF

    Args:
        X_train (pd.Series): training dataset
        X_test (pd.Series, optional): testing dataset. Defaults to None.

    Returns:
        tuple[pd.Series, pd.Series] | pd.Series: vectorized datasets
    """

    ngrange = (1, 3)
    max_feat = None

    # Initialise Tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(min_df=1, ngram_range=ngrange, max_features=max_feat)

    # fit vectorizer on train data, then apply it to test data
    X_train_vect = tfidf_vectorizer.fit_transform(X_train).toarray()

    if X_test is not None:
        X_test_vect = tfidf_vectorizer.transform(X_test).toarray()
        return X_train_vect, X_test_vect

    return X_train_vect


def split_dataset(X: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split the dataset into training and testing datasets

    Args:
        X (pd.Series): features
        y (pd.Series): labels

    Returns:
        tuple[pd.Series, pd.Series, pd.Series, pd.Series]: training and testing datasets
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

    # reset indexes
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def load_dataset(file_path: str, split: bool = True, clean: bool = True, force_reload: bool = False) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series] | tuple[pd.Series, pd.Series]:
    """Load the dataset from the file path and apply cleaning, splitting, and save the cleaned dataset to pickle files

    Args:
        file_path (str): path to the dataset file
        split (bool, optional): Split the dataset. Defaults to True.
        clean (bool, optional): Clean the dataset. Defaults to True.
        force_reload (bool, optional): Force reload the dataset, ignore pickled files. Defaults to False.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series, pd.Series] | tuple[pd.Series, pd.Series]: training and testing datasets
    """

    # check if pickle files exist
    os.makedirs('pickle', exist_ok=True)
    x_train_pickle = os.path.join('pickle/x_train.pkl')
    y_train_pickle = os.path.join('pickle/y_train.pkl')
    x_test_pickle = os.path.join('pickle/x_test.pkl')
    y_test_pickle = os.path.join('pickle/y_test.pkl')

    if pickled_data_exists([x_train_pickle, y_train_pickle, x_test_pickle, y_test_pickle]) and split and not force_reload:

        # Load pickled data
        print("Loading split dataset from pickle files")
        X_train = load_from_pickle(x_train_pickle)
        X_test = load_from_pickle(x_test_pickle)
        y_train = load_from_pickle(y_train_pickle)
        y_test = load_from_pickle(y_test_pickle)

        return X_train, X_test, y_train, y_test

    print("Loading dataset.")

    # load dataset
    data = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'])
    X, y = data["text"], data["label"]

    # clean dataset
    if clean:
        print("Cleaning dataset.")
        X_clean, y_clean = clean_dataset(X, y)
    else:
        X_clean, y_clean = X.copy(), y.copy()

    # split dataset
    if split:
        print("Splitting dataset.")
        X_train, X_test, y_train, y_test = split_dataset(X_clean, y_clean)

        # save pickle files
        save_to_pickle(x_train_pickle, X_train)
        save_to_pickle(x_test_pickle, X_test)
        save_to_pickle(y_train_pickle, y_train)
        save_to_pickle(y_test_pickle, y_test)

        return X_train, X_test, y_train, y_test

    return X_clean, y_clean
