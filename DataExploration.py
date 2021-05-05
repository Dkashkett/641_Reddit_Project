from Functions import *

X, y = import_data(PATH)

data = pd.DataFrame({"post": X, "label": y})
label_counts = data.label.value_counts().values
print(
    f"There are {label_counts[0]} non-suicidal posts and {label_counts[1]} suicidal-posts"
)
print("")

pos_posts = data[data["label"] == 1]["post"].values
neg_posts = data[data["label"] == 0]["post"].values
pos_ngram_counts = count_ngrams(pos_posts, n=2)
neg_ngram_counts = count_ngrams(neg_posts, n=2)
ngram_freq_comparison = compare_counts([pos_ngram_counts, neg_ngram_counts])

print(ngram_freq_comparison)
