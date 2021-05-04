import pandas as pd
import numpy as np
import spacy
from project_functions import *

X, y = import_data()

counts = count_ngrams(X, n=3)
print_sorted_counts(counts, n=50)

# features = convert_posts_to_features(X)
# print("finished")
