import os
import numpy as np
from data_set import filepaths as fp
import pandas as pd

base_path = fp.Ml_100K.ORGINAL_DIR
train_path = os.path.join(base_path, "ua.base")
test_path = os.path.join(base_path, "ua.test")

def get1or0(r):
    return 1.0 if r > 3 else 0.0

def __read_rating_data(path):
    triples = []
    with open(path, "r") as f:
        for line in f.readlines():
            d = line.strip().split("\t")
            triples.append([int(d[0]), int(d[1]), get1or0(int(d[2]))])
    return triples

def read_data():
    user_df = pd.read_csv(fp.Ml_100K.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.Ml_100K.ITEM_DF, index_col=0)
    train_triples = __read_rating_data(train_path)
    test_triples = __read_rating_data(test_path)
    return train_triples, test_triples, user_df, item_df, max(item_df.max()) + 1
