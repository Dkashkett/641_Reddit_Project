import pandas as pd
import numpy as np
import spacy
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from Functions import *
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import fasttext
import gensim


X, y = import_data(PATH)

# # counts = count_ngrams(X, n=3)
# # print_sorted_counts(counts, n=50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

train_features, test_features = make_features(X_train, X_test)
model = cross_validation(
    LogisticRegression(solver="liblinear"), train_features, y_train
)


# predicted_labels = lr.predict(test_features)
# accuracy = sklearn.metrics.accuracy_score(predicted_labels, y_test)
# precision = sklearn.metrics.precision_score(predicted_labels, y_test)
# recall = sklearn.metrics.recall_score(predicted_labels, y_test)

# print(accuracy)
# print("")

# skf = StratifiedKFold(n_splits=2)


# def prepare_data_for_fasttext(X, y):
#     data = pd.DataFrame(X)
#     data.columns = ["post"]
#     data["label"] = y
#     data["label"] = data["label"].map("__label__{}".format)
#     data = data[["label", "post"]]
#     return data


# data = prepare_data_for_fasttext(X_train, y_train)
# with open("./data.txt", "a") as f:
#     f.write(data.to_string(header=False, index=False))
# # counts = count_ngrams(X, n=3)
# # print_sorted_counts(counts, n=50)
