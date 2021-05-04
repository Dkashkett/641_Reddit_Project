import pandas as pd
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from collections import Counter
import re
import string
import nltk
from nltk.corpus import stopwords

# STOPWORDS_PATH = "./mallet_en_stoplist.txt"
nlp = English()
STOPWORDS = set(stopwords.words("english"))


def import_data():
    # file paths
    crowd_train_posts_path = (
        "./umd_reddit_suicidewatch_dataset_v2/crowd/train/shared_task_posts.csv"
    )
    crowd_test_posts_path = (
        "./umd_reddit_suicidewatch_dataset_v2/crowd/test/shared_task_posts_test.csv"
    )
    crowd_train_labels_path = (
        "./umd_reddit_suicidewatch_dataset_v2/crowd/train/crowd_train.csv"
    )
    crowd_test_labels_path = (
        "./umd_reddit_suicidewatch_dataset_v2/crowd/test/crowd_test.csv"
    )

    # read in files
    print("...fetching data....")
    train_posts = pd.read_csv(crowd_train_posts_path)
    train_labels = pd.read_csv(crowd_train_labels_path)
    test_posts = pd.read_csv(crowd_test_posts_path)
    test_labels = pd.read_csv(crowd_test_labels_path)

    print("...preparing dataset...")
    # fix column name for test_labels
    test_labels.columns = ["user_id", "label"]
    # merge csv into datasets for train and test
    train_data = pd.merge(train_posts, train_labels, on=["user_id"])
    test_data = pd.merge(test_posts, test_labels, on=["user_id"])
    # drop rows that have NaN values for
    train_data = train_data.dropna()
    test_data = test_data.dropna()
    # binarize labels
    train_data["label"] = train_data.label.map({"a": 0, "b": 0, "c": 0, "d": 1})
    test_data["label"] = test_data.label.map({"a": 0, "b": 0, "c": 0, "d": 1})

    # combine data
    combined_data = pd.concat([train_data, test_data])
    combined_data = combined_data.drop(
        ["post_id", "user_id", "timestamp", "subreddit", "post_title"], axis=1
    )
    combined_data.columns = ["post", "label"]

    posts = combined_data.drop(["label"], axis=1).post.values
    labels = combined_data["label"].values
    print("...imported successfully.")

    return posts, labels


def load_stopwords(filename):
    stopwords = []
    with open(filename, "r", encoding="ascii", errors="ignore") as fp:
        stopwords = fp.read().split("\n")
    return set(stopwords)


def clean_post(post):
    return re.sub("[^a-zA-Z]+", " ", post).lower()


def tokenize(post, nlp=English()):
    doc = nlp(post)
    tokens = [token.orth_.lower() for token in doc if not token.is_space]
    return tokens


def make_ngrams(tokens, n):
    return list(nltk.ngrams(tokens, n))


def filter_number_ngrams(ngrams):
    filtered = []
    for ngram in ngrams:
        condition = False
        i = 0
        while i < len(ngram):
            if ngram[i].isnumeric():
                condition = True
                i += 1
            i += 1
        if condition == False:
            filtered.append(ngram)
    return filtered


def filter_stopword_ngrams(ngrams, stopwords=STOPWORDS):
    filtered = []
    for ngram in ngrams:
        condition = False
        i = 0
        while i < len(ngram):
            if ngram[i] in stopwords:
                condition = True
                i += 1
            i += 1
        if condition == False:
            filtered.append(ngram)
    return filtered


def filter_punctuation_ngrams(ngrams):
    punct = "!!!!!!!!!!!!!\"ʖ°#$$$$$$$$$$%&'()*+,,,,,,-......./:;<=>?@[\\]^_`{|}~"
    symbols = ["-", "...", "”", "”"]
    filtered = []
    for ngram in ngrams:
        condition = False
        i = 0
        while i < len(ngram):
            if ngram[i] in punct or ngram[i] in symbols:
                condition = True
                i += 1
            i += 1
        if condition == False:
            filtered.append(ngram)
    return filtered


def filter_ngrams(ngrams, number=True, stopwords=False, punctuation=False):
    ngrams = ngrams
    if number == True:
        ngrams = filter_number_ngrams(ngrams)
    if stopwords == True:
        ngrams = filter_stopword_ngrams(ngrams)
    if punctuation == True:
        ngrams = filter_punctuation_ngrams(ngrams)

    return ["_".join(ngram) for ngram in ngrams]


def make_unigrams(post, nlp=English()):
    doc = nlp(post)
    unigrams = [
        token.orth_.lower()
        for token in doc
        if not token.is_digit
        and not token.is_punct
        and not token.is_stop
        and not token.is_space
    ]
    return unigrams


def count_ngrams(
    posts, n=2, filter_number=True, filter_stopwords=True, filter_punctuation=True
):
    d = {}
    print("...counting ngrams...")
    for post in posts:
        tokens = tokenize(post)
        ngrams = make_ngrams(tokens, n)
        filtered_ngrams = filter_ngrams(
            ngrams,
            number=filter_number,
            stopwords=filter_stopwords,
            punctuation=filter_punctuation,
        )
        counter = Counter(filtered_ngrams)
        for key, value in counter.items():
            if key in d:
                d[key] += value
            else:
                d[key] = value
    print("...finished successfully.")
    return d


def print_sorted_counts(counter, n=50, asc=False):
    frame = pd.DataFrame({"ngram": counter.keys(), "freq": counter.values()})
    sorted_frame = (
        frame.sort_values(by="freq", ascending=asc)
        .reset_index()
        .drop(["index"], axis=1)
        .head(n)
    )
    print(sorted_frame)


def convert_posts_to_features(posts, filter_stopwords=True, filter_punctuation=True):

    features = []
    for post in posts:
        tokens = tokenize(post)
        unigrams = make_unigrams(post)
        bigrams = make_ngrams(tokens, 2)
        filtered_bigrams = filter_ngrams(
            bigrams, stopwords=filter_stopwords, punctuation=filter_punctuation
        )

        feature_string = unigrams.extend(filtered_bigrams)

        features.append(feature_string)

    return features