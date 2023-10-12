import os
import joblib
import logging
import urllib3
import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm
from enum import Enum
from sklearn.decomposition import PCA

from saved_data.filepaths import Trained_para, Saved_para

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class Similarity(Enum):
    COSINE = "cosine"
    L2_NORM = "l2_norm"
    DOT_PRODUCT = "dot_product"

class IndexModel:
    index_name: str
    mapping_name: str
    dims: int
    similarity: Similarity
    
    def __init__(self, index_name, mapping_name, dims, similarity):
        self.index_name = index_name
        self.mapping_name = mapping_name
        self.dims = dims
        self.similarity = similarity

class DatasetModel:
    dataset_path: str
    columns_mappings: dict

    def __init__(self, dataset_path, columns_mappings):
        self.dataset_path = dataset_path
        self.columns_mappings = columns_mappings

class ANN_Search:
    PCA_PATH = Trained_para.PCA_PARA
    CO_MATRIX_PATH = Saved_para.CO_MATRIX

    def __init__(
        self,
        ES: object,
        index: IndexModel
    ):
        self.ES = ES
        self.index_name = index.index_name
        self.mapping_name = index.mapping_name
        self.dims = index.dims
        self.similarity = index.similarity
        self.pca = None
        self.ui_matrix = None
        self.init()

    # 创建index
    def init(self):
        self.__load_data()
        if not self.ES.indices.exists(index=self.index_name):
            self.ES.indices.create(index=self.index_name)
            self.ES.indices.put_mapping(
                index=self.index_name,
                properties={
                    self.mapping_name: {
                        "type": "dense_vector",
                        "dims": self.dims,
                        "index": True,
                        "similarity": self.similarity,
                    }
                },
            )

    # 将tsv文件数据上传到ES中
    def upload_data(self, ui_matrix):
        bulk = self.__create_bulk_from_file(ui_matrix)
        self.ES.bulk(index=self.index_name, operations=bulk)
        del bulk

    # ANN推荐
    def ann_search(self, target, k=5, num_candidates=50):
        target = self.pca.transform(np.array([target]))
        result = self.ES.knn_search(
            index=self.index_name,
            knn={
                "field": self.mapping_name,
                "query_vector": target[0],
                "k": k,
                "num_candidates": num_candidates,
            },
            source=["_id"],
        )
        return result

    # 获取U-I共现矩阵
    def get_matrix(self, dataset: DatasetModel):
        if os.path.isfile(self.CO_MATRIX_PATH):
            logging.info(f"加载本地 U-I 共现矩阵完成: {self.CO_MATRIX_PATH}")
            return np.load(self.CO_MATRIX_PATH)

        df = pd.read_csv(dataset.dataset_path, delimiter="\t", header=None).rename(columns=dataset.columns_mappings)
        users = max(df[list(dataset.columns_mappings.values())[0]].values) + 1
        items = max(df[list(dataset.columns_mappings.values())[1]].values) + 1
        ui_matrix = np.zeros((users, items))
        logging.info(f"生成 U-I 共现矩阵中...")
        for _, row in tqdm(df.iterrows()):
            ui_matrix[row.iloc[0]][row.iloc[1]] = row.iloc[2]
        np.save(file=self.CO_MATRIX_PATH, arr=ui_matrix)
        logging.info(f"共现矩阵生成完成!")
        return ui_matrix

    def train_pca(self, ui_matrix):
        if self.pca is None:
            logging.info(f"PCA参数训练中...")
            self.pca = PCA(n_components=self.dims).fit(ui_matrix)
            logging.info(f"PCA参数训练完成!")
            joblib.dump(self.pca, self.PCA_PATH)
    
    def uids_to_iids(self, ann_user_ids, ui_matrix):
        ann_user_ids = [r["_id"] for r in ann_user_ids["hits"]["hits"]]
        user_id = ann_user_ids[0]
        user_items = set(np.where(ui_matrix[int(user_id)] == 5)[0])
        other_items = [set(np.where(ui_matrix[int(id)] == 5)[0]) for id in ann_user_ids[1:]]
        recommend_items = set()
        for items in other_items:
            recommend_items |= items - user_items
        return recommend_items

    # 从tsv文件创建bulk
    def __create_bulk_from_file(self, ui_matrix):
        pca_matrix = self.pca.transform(ui_matrix)
        del ui_matrix
        iter = self.__get_iter(pca_matrix)
        bulk = [
            {"index": {"_id": f"{(_ + 1) // 2}"}} if _ % 2 == 0 else next(iter)
            for _ in range(len(pca_matrix) * 2)
        ]
        return bulk

    # 获取矩阵迭代器
    def __get_iter(self, matrix):
        for row in matrix:
            yield {self.mapping_name: list(row)}

    # 预先加载变量
    def __load_data(self):
        if os.path.isfile(self.PCA_PATH):
            self.pca = joblib.load(self.PCA_PATH)
        if os.path.isfile(self.CO_MATRIX_PATH):
            self.ui_matrix = np.load(self.CO_MATRIX_PATH)

