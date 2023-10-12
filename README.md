# recommend-system-pytorch
Recalling and Ranking with Pytorch

tree
│  README.md
│  requirements.txt
├─api
│      recommendation.py			# 推荐算法api(给一个user_id, 返回一个[item_id1, item_id2, ...])
├─data_loader
│      dataloader_ml100k.py			# 加载数据集中数据
├─data_set
│  │  filepaths.py					# 配置路径
│  ├─ml-100k
│  │      rating_index_5.tsv		# User-Item-Rating 数据集
│  └─ml-100k-orginal
│          item_df.csv				# Item特征集
│          ua.base
│          ua.test
│          user_df.csv				# User特征集
├─db
│      es.py						# ES连接配置
├─model
│      ann.py						# 召回模型
│      dcn.py						# 精排模型
└─saved_data
        co-occurrence_matrix.npy	# U-I共现矩阵变量
        dcn_model.pth				# 训练过的DCN模型
        filepaths.py				# 配置路径
        pca_parameter.pkl			# 训练过的PCA参数
