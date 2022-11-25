import os
import pandas as pd
import random


def load_file(file_name):
    df = pd.read_csv(file_name)
    actual_dict = {}
    for user, sf in df.groupby("users"):
        actual_dict[user] = list(sf["items"])

    data = df[["users", "items"]].to_numpy(dtype=int).tolist()
    return data, actual_dict, set(df["users"]), set(df["items"])


class Dataset(object):
    def __init__(self, path, name, target=True, print_summary=False):
        if target:
            self.train_path = os.path.join(path, "{}_test.csv".format(name))
            self.test_path = os.path.join(path, "{}_train.csv".format(name))
        else:
            self.train_path = os.path.join(path, "{}_train.csv".format(name))
            self.test_path = os.path.join(path, "{}_test.csv".format(name))
        self.print_summary = print_summary
        self.initialize()

    def initialize(self):
        self.train_data, self.train_dict, train_user_set, train_item_set = load_file(self.train_path)
        self.test_data, self.test_dict, test_user_set, test_item_set = load_file(self.test_path)

        # assert (test_user_set.issubset(train_user_set))
        # assert (test_item_set.issubset(train_item_set))
        self.user_set = train_user_set.union(test_user_set)
        self.item_set = train_item_set.union(test_item_set)
        self.num_user = len(self.user_set)
        self.num_item = len(self.item_set)
        self.train_size = len(self.train_data)
        self.test_size = len(self.test_data)

        self.test_input_dict, self.train_neg_dict = self.get_dicts()

        if self.print_summary:
            print("Train size:", self.train_size)
            print("Test size:", self.test_size)
            print("Number of user:", self.num_user)
            print("Number of item:", self.num_item)
            print("Data Density: {:3f}%".format(100 * self.train_size / (self.num_user * self.num_item)))

    def get_dicts(self):
        train_actual_dict, test_actual_dict = self.train_dict, self.test_dict
        train_neg_dict = {}
        test_input_dict = {}
        random.seed(0)
        for user in list(self.user_set):
            train_neg_dict[user] = list(self.item_set - set(train_actual_dict[user]))

        for user in test_actual_dict.keys():
            # test_input_dict[user] = train_neg_dict[user]
            # train_neg_dict[user] = list(set(train_neg_dict[user]) - set(test_actual_dict[user]))
            # train_neg_dict[user] = list(set(train_neg_dict[user]) - set(test_actual_dict[user]))
            random.shuffle(train_neg_dict[user])
            test_input_dict[user] = list(set(train_neg_dict[user][:len(test_actual_dict[user])*100] + test_actual_dict[user]))
            random.shuffle(test_input_dict[user])
        return test_input_dict, train_neg_dict

    def get_train(self):
        print("No training data are generated.")
        return None

    def get_data(self):
        return self.test_dict, self.test_input_dict, self.num_user, self.num_item
