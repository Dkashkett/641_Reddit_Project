import pandas as pd
import numpy as np
import spacy
from project_functions import *

X, y = import_data()

# counts = count_ngrams(X, n=3)
# print_sorted_counts(counts, n=50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 2))
# vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))

train_features = vectorizer.fit_transform(X_train)
test_features = vectorizer.transform(X_test)


lr = LogisticRegression(solver="liblinear")
lr.fit(train_features, y_train)

predicted_labels = lr.predict(test_features)
accuracy = sklearn.metrics.accuracy_score(predicted_labels, y_test)
precision = sklearn.metrics.precision_score(predicted_labels, y_test)
recall = sklearn.metrics.recall_score(predicted_labels, y_test)

print(accuracy)
print("")
