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


def make_features(X_train, X_test):
    vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 2))
    train_features = vectorizer.fit_transform(X_train)
    test_features = vectorizer.transform(X_test)
    return train_features, test_features


def run_cross_validation(
    model, train_features, y_train, metric="accuracy", n_folds=5, stratify=True
):
    if stratify == True:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=13)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=13)

    model = model
    model.fit(train_features, y_train)

    print("...running cross validation...")
    accuracy_scores = cross_val_score(
        model, train_features, y_train, scoring=metric, cv=cv
    )
    print(f"mean accuracy: {np.mean(accuracy_scores)}")
    print("")
    return model


X, y = import_data(PATH)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

train_features, test_features = make_features(X_train, X_test)
model = run_cross_validation(
    LogisticRegression(solver="liblinear"), train_features, y_train
)
