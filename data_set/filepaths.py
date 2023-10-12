import os

ROOT = os.path.split(os.path.realpath(__file__))[0]


class Ml_100K:
    __BASE = os.path.join(ROOT, "ml-100k")
    ORGINAL_DIR = os.path.join(ROOT, "ml-100k-orginal")
    USER_DF = os.path.join(ORGINAL_DIR, "user_df.csv")
    ITEM_DF = os.path.join(ORGINAL_DIR, "item_df.csv")
    RATING5 = os.path.join(__BASE, "rating_index_5.tsv")

