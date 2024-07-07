# 项目
## 天猫
### 天猫订单数据
数据集：tmall_order_report.csv

数据来源：https://www.heywhale.com/mw/dataset/6652e92372b7012cf0da08a4?shareby=667d1fb496b96d4729516592#

**项目介绍**

这个项目基于天猫订单数据进行分析和可视化，旨在了解订单情况、地域分布和时间趋势，以优化运营策略和决策支持。

**技术栈**

*Python*, *Pandas*, *Numpy*, *Pyecharts*

**项目成果**
1. 数据分析可视化
+ 分析总订单数、已完成订单数、未付款订单数、退款订单数等关键指标。
+ 使用表格和地图展示订单数据，包括各省份订单量和时间序列分析。
2. 各省订单量
+ 使用地图展示各省订单量的分布情况，支持定位主要市场和优化配送策略。
3. 时间序列分析:
+ 分析每日和每小时订单数量的趋势，识别订单高峰期，为资源分配和促销活动提供依据。
### 双十一美妆数据
数据集：双十一淘宝美妆数据.csv

数据来源：https://www.heywhale.com/mw/dataset/646343a3b1bf805682315a40?shareby=667d1fb496b96d4729516592#

**项目介绍**

这个项目基于天猫双十一美妆数据进行分析和可视化，旨在了解品牌相关数据分析和男性专用产品销售情况数据，以此优化运营策略和决策支持

**技术栈**

*Python*, *Pandas*, *Numpy*, *Pygwalker*

**项目成果**
1. 数据处理
+ 提取出有效信息形成新的特征工程
+ 建立新的特征工程——男士专用
2. 数据分析可视化
+ 分析销售量与销售额在时间维度上的变化情况
+ 针对品牌进行分析，从SKU，品牌类别占比，品牌价格角度进行分析
+ 针对各品牌销量与销售额进行分析，以品牌热度为切入点进行分析
+ 针对类别进行分析，从类别的销量与销售额，平均价格、热度与销量的关系、男性销售情况角度进行分析
3. 通过分析对促销情况进行总结，并以消费者角度对商家进行建议
## 银行客户流失的分析与预测
数据集：Customer Churn Dataset.csv

数据来源：https://www.kaggle.com/datasets/anandshaw2001/customer-churn-dataset

**项目简介**

该项目旨在通过机器学习模型预测银行客户是否会流失。客户流失是银行业中一个重要的问题，通过准确的预测模型，银行可以提前采取措施，挽留客户，减少损失。本项目从探索性数据分析，客户细分，特征重要性分析，流失预测模型四个维度展开。

**技术栈**

*Python*,*Pandas*,*Numpy*,*Matplotlib*,*Seaborn*,*Scikit-learn*

**项目成果**

1. 探索性数据分析
+ 在描述性可视化中通过客户基本信息、客户财务情况、客户信用、客户行为与偏好和流失与否的关系进行分析并得出结论
+ 在皮尔逊相关系数中，通过对正负相关性的讨论得出初步结论并提出后续的解决思路
+ 在T检验/U检验中，通过KS检验判断是否存在正态分布，T检验对服从的进行计算，其余用U检验
+ 卡方检验中判断流失组和非流失组的差异性
2. 客户细分
+ 使用聚类算法K-means进行训练，得出三个客户群体并分析三个群体的平均流失率
3. 特征重要性分析
+ 选择随机森林分类器（RandomForestClassifier）作为基线模型，使用网格搜索（GridSearchCV）进行了模型的参数调优，找到了最佳的参数组合
4. 模型评估
+ 准确率（Accuracy）：0.863，表明模型在预测客户是否流失方面有86.3%的准确率
+ 精确率（Precision）：0.776表明当模型预测客户会流失时，其预测正确的比例由77.6%。
+ 召回率（Recall）：0.425，说明在实际流失的客户中，模型能够识别出42.5%的比例。这个值相对较低，暗示模型在捕捉所有真实流失客户方面有待提升。
+ ROC曲线下面积（ROC AUC）：0.866，展现了模型在区分正负样本的整体能力，接近0.9的表明模型具有较好的分类性能。
