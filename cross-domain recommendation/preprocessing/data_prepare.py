'''extracting overlapping users'''
import pandas as pd
import numpy as np
import gzip, os


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    df = {}
    for i, d in enumerate(parse(path)):
        df[i] = d

    return pd.DataFrame.from_dict(df, orient='index')


def construct(path_s, path_t, overlapping=True):
    s_5core, t_5core = getDF(path_s), getDF(path_t)
    s_users = set(s_5core['reviewerID'].tolist())
    t_users = set(t_5core['reviewerID'].tolist())

    if overlapping:
        overlapping_users = s_users & t_users
        overlapping_users = set(list(overlapping_users)[:5000])
        s = s_5core[s_5core['reviewerID'].isin(overlapping_users)][['reviewerID', 'asin', 'overall', 'unixReviewTime']]
        t = t_5core[t_5core['reviewerID'].isin(overlapping_users)][['reviewerID', 'asin', 'overall', 'unixReviewTime']]

        csv_path_s = path_s.replace('reviews_', '').replace('_5.json.gz', '.csv').replace('data', 'data/overlapping')
        csv_path_t = path_t.replace('reviews_', '').replace('_5.json.gz', '.csv').replace('data', 'data/overlapping')
    else:
        s_users = set(np.random.choice(list(s_users-t_users), 5000, replace=False))
        t_users = set(list(t_users)[:5000])
        s = s_5core[s_5core['reviewerID'].isin(s_users)][['reviewerID', 'asin', 'overall', 'unixReviewTime']]
        t = t_5core[t_5core['reviewerID'].isin(t_users)][['reviewerID', 'asin', 'overall', 'unixReviewTime']]

        csv_path_s = path_s.replace('reviews_', '').replace('_5.json.gz', '.csv').replace('data', 'data/nonoverlapping')
        csv_path_t = path_t.replace('reviews_', '').replace('_5.json.gz', '.csv').replace('data', 'data/nonoverlapping')
    s.to_csv(csv_path_s, index=False)
    t.to_csv(csv_path_t, index=False)

    print('Build raw data to %s.' % csv_path_s)
    print('Build raw data to %s.' % csv_path_t)


if __name__ == '__main__':
    construct('data/reviews_Movies_and_TV_5.json.gz', 'data/reviews_Books_5.json.gz')
    construct('data/reviews_CDs_and_Vinyl_5.json.gz', 'data/reviews_Digital_Music_5.json.gz')

    construct('data/reviews_Movies_and_TV_5.json.gz', 'data/reviews_Books_5.json.gz', overlapping=False)
    construct('data/reviews_CDs_and_Vinyl_5.json.gz', 'data/reviews_Digital_Music_5.json.gz', overlapping=False)

