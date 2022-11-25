import os
import numpy as np
import pandas as pd
from itertools import count
from collections import defaultdict
from multiprocessing import Pool


class Dataset:
    def __init__(self, file_path, name, folder_name, dir_root):
        self.processor_num = 12
        self.test_size = 1
        self.path = '/'.join(file_path.split('/')[:-1])
        self.name = name
        self.folder_name = folder_name
        self.dir_root = dir_root
        self._bulid_data(file_path)

    def _bulid_data(self, file_path):
        df, self.num_users, self.num_items = self._load_data(file_path)
        self.train_df, self.test_df = self._split_train_test(df)

    def _load_data(self, file_path):
        df = pd.read_csv(file_path, sep=',', usecols=[0, 1])

        # constructing index
        uiterator = count(0)
        udict = defaultdict(lambda: next(uiterator))
        [udict[user] for user in sorted(df['reviewerID'].tolist())]
        iiterator = count(0)
        idict = defaultdict(lambda: next(iiterator))
        [idict[item] for item in sorted(df['asin'].tolist())]

        self.udict = udict
        self.idict = idict

        df['users'] = df['reviewerID'].map(lambda x: udict[x])
        df['items'] = df['asin'].map(lambda x: idict[x])
        del df['reviewerID'], df['asin']
        print('Load %s data successfully with %d users, %d products and %d interactions.'
              % (self.name, len(udict), len(idict), df.shape[0]))

        return df, len(udict), len(idict)

    def _split_train_test(self, df):
        print('Spliting data of train and test...')
        with Pool(self.processor_num) as pool:
            nargs = [(user, df, self.test_size) for user in range(self.num_users)]
            test_list = pool.map(self._split, nargs)

        test_df = pd.concat(test_list)
        train_df = df.drop(test_df.index)

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df.to_csv('{}/{}/{}_train.csv'.format(self.dir_root, self.folder_name, self.name), index=False)
        test_df.to_csv('{}/{}/{}_test.csv'.format(self.dir_root, self.folder_name, self.name), index=False)

        return train_df, test_df

    @staticmethod
    def _split(args):
        user, df, test_size = args
        sample_test = df[df['users'] == user].sample(n=test_size)

        return sample_test


if __name__ == '__main__':
    dir_root = "data/overlapping"
    folder_name = "CD_Music"
    Dataset(dir_root+'/CDs_and_Vinyl.csv', name='cd', folder_name=folder_name, dir_root=dir_root)
    Dataset(dir_root+'/Digital_Music.csv', name='music', folder_name=folder_name, dir_root=dir_root)

    folder_name = "Book_Movie"
    Dataset(dir_root + '/Books.csv', name='book', folder_name=folder_name, dir_root=dir_root)
    Dataset(dir_root + '/Movies_and_TV.csv', name='movie', folder_name=folder_name, dir_root=dir_root)
