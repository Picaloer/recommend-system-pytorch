import numpy as np
from typing import List

import db.es as es
from model.ann import ANN_Search, IndexModel, Similarity, DatasetModel
from data_set.filepaths import Ml_100K
from model.dcn import DCN_Model, DCN_Ranking
import data_loader.dataloader_ml100k as dataloader_ml100k

ES = es.init_ES()

dataset = DatasetModel(
    dataset_path=Ml_100K.RATING5,
    columns_mappings={0: "user_id", 1: "item_id", 2: "rating"}
)

index = IndexModel(
    index_name="ann_04",
    mapping_name="uir",
    dims=32,
    similarity=Similarity.COSINE,
)

def recall_uui(id: int, k: int = 10, update_data: bool = True) -> List[int]:
    # 1. 创建ANNSearch索引, 默认向量维度512, 相似度公式cosine
    ann = ANN_Search(ES, index)

    # 2. 获取 U-I 共现矩阵
    matrix = ann.get_matrix(dataset)

    # 3. 训练PCA权重
    ann.train_pca(matrix)

    # 4. PCA降维, 结果上传至ES
    if update_data:
        ann.upload_data(matrix)

    # 5. 输入原始特征, 获取k个最近邻User, 返回user_id列表
    recommend_user_ids = ann.ann_search(matrix[id], k)

    # 6. 获取推荐的item_id列表
    recommend_items_ids = ann.uids_to_iids(recommend_user_ids, matrix)

    return recommend_items_ids


def recall_uii(ES, iu_vector, k, n):
    return


def rank(id: int, recommendations: List[int], if_train: bool) -> List[int]:
    # 1. 加载数据
    (
        train_triples,
        test_triples,
        user_df,
        item_df,
        n_features,
    ) = dataloader_ml100k.read_data()

    # 2. 创建DCN模型
    dcn = DCN_Model(n_features, user_df, item_df)

    # 3. 创建DCNRanking模型
    dcn_ranking = DCN_Ranking(dcn)

    # 4. 训练模型
    if if_train:
        dcn_ranking.train(train_triples)

    # 5. 模型预测
    ui_ids = [(id, item_id) for item_id in recommendations]
    preds_result = dcn_ranking.pred(ui_ids)

    # 6. 获取推荐列表
    ranking_list = dcn_ranking.rank_by_score(recommendations, preds_result)

    return ranking_list


def recommend(user_id: int, if_train: bool = True):
    recommendations = rank(user_id, recall_uui(user_id), if_train)
    return recommendations

if __name__ == '__main__':
    print(recommend(user_id = 1, if_train = True))