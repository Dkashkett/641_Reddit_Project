
def create_dataset_and_directory(X, y, destination_path, test_size=0.1):
    dir_path = os.path.join(destination_path, "dataset")
    if os.path.isdir(dir_path) == True:
        print("dataset already created.")
        return
    else:
        print("...creating directories...")

        train_path = os.path.join(dir_path, "train")
        test_path = os.path.join(dir_path, "test")
        train_pos_path = os.path.join(train_path, "suicidal")
        train_neg_path = os.path.join(train_path, "non_suicidal")
        test_pos_path = os.path.join(test_path, "suicidal")
        test_neg_path = os.path.join(test_path, "non_suicidal")
        os.mkdir(dir_path)
        os.mkdir(train_path)
        os.mkdir(test_path)
        os.mkdir(train_pos_path)
        os.mkdir(train_neg_path)
        os.mkdir(test_pos_path)
        os.mkdir(test_neg_path)
        print("...splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=13
        )
        # train
        print("...processing training data...")
        data = pd.DataFrame({"post": X_train, "label": y_train})
        pos_posts = data[data["label"] == 1]["post"].values
        neg_posts = data[data["label"] == 0]["post"].values
        for index, post in enumerate(pos_posts):
            post_path = os.path.join(train_pos_path, str(index))
            with open(post_path, "wt") as f:
                f.write(post)
                f.close()
        for index, post in enumerate(neg_posts):
            post_path = os.path.join(train_neg_path, str(index))
            with open(post_path, "wt") as f:
                f.write(post)
                f.close()
        # test
        print("...processing test data...")
        data = pd.DataFrame({"post": X_test, "label": y_test})
        pos_posts = data[data["label"] == 1]["post"].values
        neg_posts = data[data["label"] == 0]["post"].values
        for index, post in enumerate(pos_posts):
            post_path = os.path.join(test_pos_path, str(index))
            with open(post_path, "wt") as f:
                f.write(post)
                f.close()
        for index, post in enumerate(neg_posts):
            post_path = os.path.join(test_neg_path, str(index))
            with open(post_path, "wt") as f:
                f.write(post)
                f.close()
        print("...dataset and directories created succesfully.")

