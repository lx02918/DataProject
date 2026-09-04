# DataProject — 数据分析与建模项目集

个人数据分析学习与实践仓库，涵盖探索性分析、统计检验、机器学习建模、推荐系统与可视化。数据均来自 Kaggle、和鲸社区等公开数据集。

> 作者：周小梵 · [个人主页](https://www.lx02918.ltd) · 数据科学与大数据技术

---

## 数据分析项目

| 项目 | 内容 | 技术栈 | 关键产出 |
| --- | --- | --- | --- |
| [银行客户流失分析与预测](./Bank_Customer_Churn/) | EDA → 假设检验（KS/T/U/卡方）→ KMeans 客户细分 → 随机森林流失预测（GridSearch 调参） | Python, Pandas, Scikit-learn, Seaborn | 准确率 0.863，ROC AUC 0.866；输出 3 类客户群及差异化挽留思路 |
| [客户购物趋势分析](./Shopping_Trend_Analysis/) | 多维度消费行为分析、客户忠诚度分层、基于 SVD 的商品推荐 | Python, Pandas, Scikit-learn, SciPy | 完整分析报告 + 可运行的用户画像推荐 demo |
| [天猫订单 & 双十一美妆数据分析](./天猫/) | 订单关键指标、各省订单量地图、时间序列高峰识别；美妆品牌/类别/价格多角度分析 | Python, Pandas, Pyecharts, Pygwalker | 交互式可视化 + 面向运营与消费者的建议 |
| [纽约出租车行程时长预测](./New%20York%20Taxi/) | 特征工程（时间、距离、天气、节假日）+ 梯度提升回归 | Python, XGBoost, Pandas, GeoJSON | Kaggle 赛题完整建模流程 |
| [社交网络用户分析](./Social%20Network/) | 中文分词与词频统计、KMeans 用户标签聚类、网络关系可视化 | Python, jieba, Scikit-learn, pyecharts | 用户标签体系 + 关系网络图（render.html） |
| [电影推荐系统](./Movie_Recommend/) | 基于协同过滤（UserCF / ItemCF）的推荐，Django 前后端 + MySQL，支持注册/登录/评分/推荐 | Python, Django, MySQL, jQuery | 可交互 Web 应用 |

## 学习笔记

| 目录 | 内容 |
| --- | --- |
| [leetcode](./leetcode/) | 按专题整理的刷题代码与思路（数组、哈希、双指针、滑动窗口、二叉树、图、回溯、动态规划等） |
| [pytorch](./pytorch/) | PyTorch 入门课程代码与笔记 |
| [笔试记录](./笔试记录/) | 各公司数据岗笔试题的解题记录 |

---

## 说明

- 部分数据集体积较大未纳入版本库，数据来源均在各子项目 README 中标注，可自行下载复现。
- 项目为学习实践性质，欢迎交流指正。
