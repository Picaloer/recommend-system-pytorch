import os
import numpy as np
from typing import List
import torch
from torch import nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score
from saved_data.filepaths import Trained_para


class CrossNet(nn.Module):
    def __init__(self, in_features, layer_num=2, seed=1024):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.kernels = torch.nn.ParameterList(
            [
                nn.Parameter(nn.init.xavier_normal_(torch.empty(in_features, 1)))
                for i in range(self.layer_num)
            ]
        )
        self.bias = torch.nn.ParameterList(
            [
                nn.Parameter(nn.init.zeros_(torch.empty(in_features, 1)))
                for i in range(self.layer_num)
            ]
        )

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)  # [1024, 10, 1, 128]
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = torch.tensordot(
                x_l, self.kernels[i], dims=([1], [0])
            )  # [1024, 10, 128, 1]
            dot_ = torch.matmul(x_0, xl_w)  # [1024, 10, 1, 1]
            x_l = dot_ + self.bias[i] + x_l  # [1024, 10, 10, 128]
        x_l = torch.sum(x_l.reshape(x_l.shape[0], -1), dim=1) / (10 * 10 * 128)
        x_l = torch.squeeze(x_l)
        return x_l


class DCN_Model(nn.Module):
    def __init__(self, n_features, user_df, item_df, dim=128, num_cross_layers=3):
        super(DCN_Model, self).__init__()
        # 随机初始化所有特征的特征向量
        self.features = nn.Embedding(n_features, dim, max_norm=1)
        # 记录好用户和物品的特征索引
        self.user_df = user_df
        self.item_df = item_df
        # 得到用户和物品特征的数量的和
        total_neigbours = user_df.shape[1] + item_df.shape[1]
        # 初始化MLP层
        self.mlp_layer = self.__mlp(dim * total_neigbours)
        self.cross_net = CrossNet(total_neigbours)

    def __mlp(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    # #Cross部分(被CrossNet替换)
    # def Cross( self, feature_embs, num_cross_layers = 2 ):
    #     # feature_embs:[ batch_size, n_features, dim ]
    #     cross = nn.ModuleList()
    #     for _ in range(num_cross_layers):
    #         cross.append(nn.Linear(feature_embs.shape[1], feature_embs.shape[1]))

    #     x0 = feature_embs
    #     x = x0
    #     for i in range(num_cross_layers):
    #         x = torch.matmul(x.unsqueeze(3), x.unsqueeze(2))

    #     # x:[ batch_size, n_features, dim ] -> [batch_size]
    #     return x
    # DNN部分
    def Deep(self, feature_embs):
        # feature_embs:[ batch_size, n_features, dim ]
        # [ batch_size, total_neigbours * dim ]
        feature_embs = feature_embs.reshape((feature_embs.shape[0], -1))
        # [ batch_size, 1 ]
        output = self.mlp_layer(feature_embs)
        # [ batch_size ]
        return torch.squeeze(output)

    # 把用户和物品的特征合并起来
    def __getAllFeatures(self, u, i):
        users = torch.LongTensor(self.user_df.loc[u].values)
        items = torch.LongTensor(self.item_df.loc[i].values)
        all = torch.cat([users, items], dim=1)
        return all

    # 前向传播方法
    def forward(self, u, i):
        # 得到用户与物品组合起来后的特征索引
        all_feature_index = self.__getAllFeatures(u, i)
        # print(all_feature_index.shape)
        # 取出特征向量
        all_feature_embs = self.features(all_feature_index)
        # [batch_size]
        fm_out = self.cross_net(all_feature_embs)
        # [batch_size]
        deep_out = self.Deep(all_feature_embs)
        # [batch_size]
        out = torch.sigmoid(fm_out + deep_out)
        return out


# DCN精排
class DCN_Ranking:
    MODEL_PATH = Trained_para.DCN_MODEL

    def __init__(self, net: DCN_Model):
        self.net = net

    def pred(self, input_features) -> List[float]:
        if not os.path.exists(self.MODEL_PATH):
            logging.info("没有找到可用的训练模型...")
            return None
        if os.path.exists(self.MODEL_PATH):
            net = torch.load(self.MODEL_PATH)
            net.eval()
            logits = []
            for u, i in DataLoader(input_features):
                logits.append(net(u, i).item())
        return logits

    def train(self, train_triples, epochs=20, batch_size=1024, lr=0.01, dim=128):
        # 训练模型
        logging.info("DCN模型开始训练...")
        self.__train_model(
            net=self.net,
            train_triples=train_triples,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )
        logging.info("DCN模型训练完成!")
    
    def rank_by_score(self, recommendations, preds_result):
        scores_dict = {
            item_id: score for item_id, score in zip(recommendations, preds_result)
        }
        scores_dict = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        return list(dict(scores_dict).keys())

    # 做评估
    def doEva(self, net, test_triple):
        d = torch.LongTensor(test_triple)
        u, i, r = d[:, 0], d[:, 1], d[:, 2]
        with torch.no_grad():
            out = net(u, i)
        y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
        precision = precision_score(r, y_pred)
        recall = recall_score(r, y_pred)
        acc = accuracy_score(r, y_pred)
        return precision, recall, acc

    def __train_model(
        self,
        net,
        train_triples,
        epochs,
        batch_size,
        lr,
        test_triples=None,
        eva_per_epochs=0,
    ):
        # 定义损失函数
        criterion = torch.nn.BCELoss()
        # 初始化优化器
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
        # 开始训练
        for e in tqdm(range(epochs)):
            all_lose = 0
            for u, i, r in DataLoader(
                train_triples, batch_size=batch_size, shuffle=True
            ):
                r = torch.FloatTensor(r.detach().numpy())
                optimizer.zero_grad()
                logits = net(u, i)
                loss = criterion(logits, r)
                all_lose += loss
                loss.backward()
                optimizer.step()
            # 评估模型
            if eva_per_epochs != 0 and e % eva_per_epochs == 0:
                logging.info(
                    "DCN模型训练中...epoch {},avg_loss={:.4f}".format(
                        e, all_lose / (len(train_triples) // batch_size)
                    )
                )
                p, r, acc = self.doEva(net, train_triples)
                logging.info(
                    "DCN模型训练中...train:p:{:.4f}, r:{:.4f}, acc:{:.4f}".format(p, r, acc)
                )
                p, r, acc = self.doEva(net, test_triples)
                logging.info(
                    "DCN模型训练中...test:p:{:.4f}, r:{:.4f}, acc:{:.4f}".format(p, r, acc)
                )
        torch.save(net, self.MODEL_PATH)
