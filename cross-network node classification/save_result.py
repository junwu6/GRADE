import pandas as pd


def save_to_csv(results, save_name=None):
    df = pd.DataFrame(results)
    df.to_csv("{}.csv".format(save_name))
