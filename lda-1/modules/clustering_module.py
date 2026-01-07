# -*- coding: utf-8 -*-
"""
文本聚类与分类模块 (Text Clustering and Classification Module)
==============================================================

本模块提供完整的文本聚类和分类功能，是文本分析系统的核心组件之一。

模块概述
--------
文本聚类是一种无监督学习方法，用于将相似的文档自动分组，无需预先定义类别。
文本分类是一种半监督学习方法，通过少量人工标注数据训练模型，自动分类剩余文档。

主要功能
--------
1. **文本聚类 (Text Clustering)**
   - K-means聚类：基于质心的划分聚类算法
   - 层次聚类：基于距离的凝聚聚类算法
   - 聚类关键词提取：自动识别每个聚类的代表性词汇
   - 聚类可视化：t-SNE降维散点图、层次聚类树状图

2. **文本分类 (Text Classification)**
   - 自定义标签管理：创建、删除分类标签
   - 手动标注：为文档分配分类标签
   - 自动分类：基于KNN算法预测未标注文档的类别

3. **文档向量化 (Document Vectorization)**
   - TF-IDF向量化：基于词频-逆文档频率的文档表示
   - 主题分布向量化：基于LDA主题模型的文档表示

核心算法说明
------------

### K-means聚类算法
K-means是一种迭代优化算法，目标是最小化簇内平方和(Within-Cluster Sum of Squares, WCSS)。

算法步骤：
1. 随机初始化K个聚类中心
2. 将每个数据点分配到最近的聚类中心
3. 重新计算每个聚类的中心（取均值）
4. 重复步骤2-3直到收敛

数学表达：
- 目标函数: J = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
- 其中 Cᵢ 是第i个聚类，μᵢ 是聚类中心

优点：简单高效，适合大规模数据
缺点：需要预先指定K值，对初始化敏感，假设聚类为球形

### 层次聚类算法 (Agglomerative Clustering)
层次聚类是一种自底向上的凝聚聚类方法，逐步合并最相似的聚类。

链接方法 (Linkage Methods)：
- Ward: 最小化合并后的总方差增量，倾向于产生大小相近的聚类
- Complete: 使用两个聚类间的最大距离，产生紧凑的聚类
- Average: 使用两个聚类间的平均距离，介于Ward和Complete之间
- Single: 使用两个聚类间的最小距离，可能产生链状聚类

算法步骤：
1. 将每个数据点视为一个独立的聚类
2. 计算所有聚类对之间的距离
3. 合并距离最近的两个聚类
4. 重复步骤2-3直到达到指定的聚类数量

### TF-IDF向量化
TF-IDF (Term Frequency-Inverse Document Frequency) 是一种文本特征提取方法。

计算公式：
- TF(t,d) = 词t在文档d中出现的次数 / 文档d的总词数
- IDF(t) = log(文档总数 / 包含词t的文档数)
- TF-IDF(t,d) = TF(t,d) × IDF(t)

特点：
- 高TF-IDF值表示词在当前文档中重要但在整个语料库中不常见
- 有效降低常见词的权重，突出文档的特征词

### t-SNE降维算法
t-SNE (t-distributed Stochastic Neighbor Embedding) 是一种非线性降维技术，
特别适合高维数据的可视化。

核心思想：
1. 在高维空间中计算数据点之间的相似度（使用高斯分布）
2. 在低维空间中计算数据点之间的相似度（使用t分布）
3. 最小化两个分布之间的KL散度

参数说明：
- perplexity: 控制局部与全局结构的平衡，通常设置为5-50

### KNN分类算法
K-Nearest Neighbors (KNN) 是一种基于实例的学习算法。

算法原理：
1. 对于待分类样本，找到训练集中距离最近的K个样本
2. 统计这K个样本的类别分布
3. 将待分类样本归为出现次数最多的类别

距离度量：通常使用欧氏距离或余弦相似度

使用示例
--------
```python
# 创建文档向量
doc_vectors = create_doc_vectors(texts, method='tfidf')

# K-means聚类
clusterer = TextClusterer(doc_vectors, file_names)
labels = clusterer.kmeans_clustering(n_clusters=5)

# 获取聚类关键词
keywords = clusterer.get_all_cluster_keywords(texts, top_n=10)

# 文本分类
classifier = TextClassifier(doc_vectors, file_names)
classifier.add_label_category("正面")
classifier.add_label_category("负面")
classifier.add_label("doc1.txt", "正面")
classifier.train_classifier()
predictions = classifier.predict_unlabeled()
```

Requirements: 3.1-3.7

作者: Text Analysis System
版本: 2.0
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime
import os

from utils.session_state import log_message


class TextClusterer:
    """
    文本聚类器 (Text Clusterer)
    ===========================
    
    提供文档自动聚类功能，支持K-means和层次聚类两种主流算法。
    聚类是一种无监督学习方法，无需预先定义类别即可发现文档的自然分组。
    
    核心功能
    --------
    1. K-means聚类：快速划分聚类，适合大规模数据
    2. 层次聚类：构建层次结构，可生成树状图
    3. 聚类关键词提取：识别每个聚类的代表性词汇
    4. 降维可视化：使用t-SNE将高维向量映射到2D空间
    
    属性说明
    --------
    doc_vectors : np.ndarray
        文档向量矩阵，形状为 (n_docs, n_features)
        每行代表一个文档的特征向量
    file_names : List[str]
        文档名称列表，与doc_vectors的行一一对应
    cluster_labels : np.ndarray
        聚类标签数组，每个元素表示对应文档所属的聚类ID
    cluster_centers : np.ndarray
        聚类中心矩阵（仅K-means），形状为 (n_clusters, n_features)
    linkage_matrix : np.ndarray
        链接矩阵（仅层次聚类），用于绘制树状图
    n_clusters : int
        聚类数量
    algorithm : str
        当前使用的聚类算法名称 ('kmeans' 或 'hierarchical')
    
    使用示例
    --------
    >>> # 创建聚类器
    >>> clusterer = TextClusterer(doc_vectors, file_names)
    >>> 
    >>> # 执行K-means聚类
    >>> labels = clusterer.kmeans_clustering(n_clusters=5)
    >>> 
    >>> # 获取聚类摘要
    >>> summary = clusterer.get_cluster_summary(texts)
    
    Requirements: 3.1, 3.2, 3.4
    """
    
    def __init__(self, doc_vectors: np.ndarray, file_names: List[str]):
        """
        初始化聚类器
        
        创建一个新的TextClusterer实例，准备对给定的文档向量进行聚类分析。
        
        参数说明
        --------
        doc_vectors : np.ndarray
            文档向量矩阵，形状为 (n_docs, n_features)
            - n_docs: 文档数量
            - n_features: 特征维度（如TF-IDF的词汇表大小）
            
        file_names : List[str]
            文档名称列表，长度必须等于n_docs
            用于在结果中标识每个文档
        
        注意事项
        --------
        - doc_vectors应该已经过预处理（如TF-IDF向量化）
        - 建议在聚类前对向量进行标准化处理
        """
        self.doc_vectors = doc_vectors
        self.file_names = file_names
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.linkage_matrix: Optional[np.ndarray] = None
        self.n_clusters: int = 0
        self.algorithm: str = ""
    
    def kmeans_clustering(self, n_clusters: int, random_state: int = 42) -> np.ndarray:
        """
        K-means聚类算法
        ===============
        
        使用K-means算法将文档划分为K个互不重叠的聚类。
        
        算法原理
        --------
        K-means是一种基于质心的划分聚类算法，通过迭代优化最小化
        簇内平方和(WCSS)来找到最优的聚类划分。
        
        算法步骤：
        1. **初始化**: 随机选择K个数据点作为初始聚类中心
        2. **分配**: 将每个数据点分配到距离最近的聚类中心
        3. **更新**: 重新计算每个聚类的中心（取聚类内所有点的均值）
        4. **迭代**: 重复步骤2-3直到聚类中心不再变化或达到最大迭代次数
        
        数学表达：
        - 目标函数: J = Σᵢ₌₁ᴷ Σₓ∈Cᵢ ||x - μᵢ||²
        - 其中 Cᵢ 是第i个聚类，μᵢ 是聚类中心
        
        参数说明
        --------
        n_clusters : int
            聚类数量K，决定将文档分成多少组
            - 建议范围: 2 到 √(n/2)，其中n是文档数量
            - 可使用肘部法则(Elbow Method)确定最优K值
            
        random_state : int, default=42
            随机种子，用于初始化聚类中心
            - 设置固定值可确保结果可重现
            - 不同的随机种子可能产生不同的聚类结果
        
        返回值
        ------
        np.ndarray
            聚类标签数组，形状为 (n_docs,)
            - 每个元素的值范围是 [0, n_clusters-1]
            - 相同标签的文档属于同一个聚类
        
        算法特点
        --------
        优点：
        - 计算效率高，时间复杂度为 O(n*K*I*d)
          其中n是样本数，K是聚类数，I是迭代次数，d是特征维度
        - 结果易于解释，聚类中心可作为聚类的代表
        - 适合处理大规模数据集
        
        缺点：
        - 需要预先指定聚类数量K
        - 对初始化敏感，可能陷入局部最优
        - 假设聚类为球形，对非凸形状的聚类效果较差
        - 对异常值敏感
        
        使用建议
        --------
        - 当文档数量较大(>1000)时优先选择K-means
        - 如果不确定K值，可以尝试多个值并比较轮廓系数
        - 对于文本数据，通常K值在3-10之间效果较好
        
        示例
        ----
        >>> clusterer = TextClusterer(doc_vectors, file_names)
        >>> labels = clusterer.kmeans_clustering(n_clusters=5)
        >>> print(f"聚类0包含 {sum(labels==0)} 个文档")
        """
        # 处理边界情况：聚类数不能超过文档数
        if len(self.doc_vectors) < n_clusters:
            n_clusters = len(self.doc_vectors)
        
        # 创建K-means模型
        # n_init=10: 使用10次不同的初始化，选择最优结果
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        
        # 执行聚类并获取标签
        self.cluster_labels = kmeans.fit_predict(self.doc_vectors)
        
        # 保存聚类中心，可用于分析聚类特征
        self.cluster_centers = kmeans.cluster_centers_
        
        self.n_clusters = n_clusters
        self.algorithm = "kmeans"
        
        return self.cluster_labels
    
    def hierarchical_clustering(self, n_clusters: int, linkage_method: str = 'ward') -> np.ndarray:
        """
        层次聚类算法 (Agglomerative Hierarchical Clustering)
        ====================================================
        
        使用凝聚层次聚类算法构建文档的层次结构，并划分为指定数量的聚类。
        
        算法原理
        --------
        层次聚类是一种自底向上(Bottom-up)的凝聚聚类方法：
        1. **初始化**: 将每个文档视为一个独立的聚类
        2. **计算距离**: 计算所有聚类对之间的距离
        3. **合并**: 找到距离最近的两个聚类并合并
        4. **迭代**: 重复步骤2-3直到达到指定的聚类数量
        
        该过程构建一棵树状结构(Dendrogram)，可以在任意层级切割得到不同数量的聚类。
        
        参数说明
        --------
        n_clusters : int
            目标聚类数量
            - 决定在树状图的哪个高度进行切割
            
        linkage_method : str, default='ward'
            链接方法，决定如何计算聚类间的距离
            
            可选值及其特点：
            
            - **'ward'** (默认推荐)
              Ward方差最小化方法
              - 原理: 最小化合并后聚类内方差的增量
              - 特点: 倾向于产生大小相近、紧凑的聚类
              - 适用: 大多数场景的首选方法
            
            - **'complete'** (最远距离法)
              - 原理: 使用两个聚类中最远点对之间的距离
              - 特点: 产生紧凑的聚类，对异常值敏感
              - 适用: 希望聚类内部紧密时使用
            
            - **'average'** (平均距离法/UPGMA)
              - 原理: 使用两个聚类中所有点对距离的平均值
              - 特点: 介于ward和complete之间，较为稳健
              - 适用: 数据分布不均匀时
            
            - **'single'** (最近距离法)
              - 原理: 使用两个聚类中最近点对之间的距离
              - 特点: 可能产生链状聚类(chaining effect)
              - 适用: 检测细长形状的聚类
        
        返回值
        ------
        np.ndarray
            聚类标签数组，形状为 (n_docs,)
        
        算法特点
        --------
        优点：
        - 不需要预先指定聚类数量（可以事后决定）
        - 产生层次结构，便于多粒度分析
        - 可以生成树状图进行可视化
        - 对聚类形状没有假设
        
        缺点：
        - 时间复杂度较高 O(n³)，不适合大规模数据
        - 一旦合并无法撤销（贪心算法）
        - 对噪声和异常值敏感
        
        使用建议
        --------
        - 文档数量较少(<500)时推荐使用
        - 需要可视化聚类层次结构时使用
        - 不确定最佳聚类数量时，可以先聚类再根据树状图决定
        
        示例
        ----
        >>> clusterer = TextClusterer(doc_vectors, file_names)
        >>> labels = clusterer.hierarchical_clustering(n_clusters=4, linkage_method='ward')
        """
        # 处理边界情况
        if len(self.doc_vectors) < n_clusters:
            n_clusters = len(self.doc_vectors)
        
        # 创建凝聚聚类模型
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        
        # 执行聚类
        self.cluster_labels = clustering.fit_predict(self.doc_vectors)
        self.n_clusters = n_clusters
        self.algorithm = "hierarchical"
        
        # 计算链接矩阵用于绘制树状图
        # linkage函数返回一个(n-1, 4)的矩阵，每行表示一次合并操作
        # 列含义: [聚类1索引, 聚类2索引, 距离, 新聚类包含的样本数]
        self.linkage_matrix = linkage(self.doc_vectors, method=linkage_method)
        
        return self.cluster_labels

    def get_cluster_keywords(self, cluster_id: int, texts: List[List[str]], top_n: int = 10) -> List[str]:
        """
        提取聚类代表性关键词
        ====================
        
        通过词频统计提取指定聚类中最具代表性的关键词。
        
        算法原理
        --------
        使用简单的词频统计方法：
        1. 收集属于该聚类的所有文档
        2. 统计这些文档中所有词的出现频率
        3. 返回频率最高的top_n个词作为关键词
        
        这种方法假设高频词能够代表聚类的主题特征。
        
        参数说明
        --------
        cluster_id : int
            聚类ID，范围 [0, n_clusters-1]
            
        texts : List[List[str]]
            分词后的文本列表
            - 外层列表对应每个文档
            - 内层列表是该文档的词语序列
            
        top_n : int, default=10
            返回的关键词数量
        
        返回值
        ------
        List[str]
            关键词列表，按频率降序排列
        
        注意事项
        --------
        - 建议在分词时已去除停用词，否则关键词可能包含无意义的高频词
        - 对于小聚类，返回的关键词数量可能少于top_n
        
        示例
        ----
        >>> keywords = clusterer.get_cluster_keywords(0, texts, top_n=5)
        >>> print(f"聚类0的关键词: {keywords}")
        """
        if self.cluster_labels is None:
            return []
        
        # 获取属于该聚类的文档索引
        cluster_doc_indices = np.where(self.cluster_labels == cluster_id)[0]
        
        if len(cluster_doc_indices) == 0:
            return []
        
        # 统计该聚类中所有词的频率
        word_counter = Counter()
        for idx in cluster_doc_indices:
            if idx < len(texts):
                word_counter.update(texts[idx])
        
        # 返回最常见的词
        return [word for word, _ in word_counter.most_common(top_n)]
    
    def get_all_cluster_keywords(self, texts: List[List[str]], top_n: int = 10) -> Dict[int, List[str]]:
        """
        获取所有聚类的代表性关键词
        ==========================
        
        批量提取每个聚类的关键词，便于整体分析聚类特征。
        
        参数说明
        --------
        texts : List[List[str]]
            分词后的文本列表
            
        top_n : int, default=10
            每个聚类返回的关键词数量
        
        返回值
        ------
        Dict[int, List[str]]
            聚类ID到关键词列表的映射
            - 键: 聚类ID (0 到 n_clusters-1)
            - 值: 该聚类的关键词列表
        
        示例
        ----
        >>> all_keywords = clusterer.get_all_cluster_keywords(texts)
        >>> for cluster_id, keywords in all_keywords.items():
        ...     print(f"聚类{cluster_id}: {', '.join(keywords[:5])}")
        """
        if self.cluster_labels is None:
            return {}
        
        keywords = {}
        for cluster_id in range(self.n_clusters):
            keywords[cluster_id] = self.get_cluster_keywords(cluster_id, texts, top_n)
        
        return keywords
    
    def get_cluster_documents(self, cluster_id: int) -> List[str]:
        """
        获取指定聚类中的文档名称
        
        参数说明
        --------
        cluster_id : int
            聚类ID，范围 [0, n_clusters-1]
        
        返回值
        ------
        List[str]
            属于该聚类的文档名称列表
        """
        if self.cluster_labels is None:
            return []
        
        cluster_doc_indices = np.where(self.cluster_labels == cluster_id)[0]
        return [self.file_names[i] for i in cluster_doc_indices if i < len(self.file_names)]
    
    def get_cluster_summary(self, texts: List[List[str]], top_n: int = 10) -> pd.DataFrame:
        """
        生成聚类摘要报告
        ================
        
        汇总所有聚类的关键信息，包括文档数量、文档列表和代表性关键词。
        
        参数说明
        --------
        texts : List[List[str]]
            分词后的文本列表
            
        top_n : int, default=10
            每个聚类提取的关键词数量
        
        返回值
        ------
        pd.DataFrame
            聚类摘要表格，包含以下列：
            - 聚类ID: 聚类编号（从1开始，便于用户理解）
            - 文档数量: 该聚类包含的文档数
            - 文档列表: 前5个文档名称（超过5个显示省略号）
            - 代表性关键词: 前5个关键词
        
        示例
        ----
        >>> summary = clusterer.get_cluster_summary(texts)
        >>> print(summary.to_string())
        """
        if self.cluster_labels is None:
            return pd.DataFrame()
        
        summary_data = []
        for cluster_id in range(self.n_clusters):
            docs = self.get_cluster_documents(cluster_id)
            keywords = self.get_cluster_keywords(cluster_id, texts, top_n)
            
            summary_data.append({
                '聚类ID': cluster_id + 1,
                '文档数量': len(docs),
                '文档列表': ', '.join(docs[:5]) + ('...' if len(docs) > 5 else ''),
                '代表性关键词': ', '.join(keywords[:5])
            })
        
        return pd.DataFrame(summary_data)
    
    def export_results(self) -> str:
        """
        导出聚类结果为CSV格式
        
        将聚类结果导出为CSV格式字符串，便于保存和分享。
        
        返回值
        ------
        str
            CSV格式的聚类结果字符串
            包含列: 文档名, 聚类ID（从1开始）
        """
        if self.cluster_labels is None:
            return ""
        
        data = []
        for i, (name, label) in enumerate(zip(self.file_names, self.cluster_labels)):
            data.append({
                '文档名': name,
                '聚类ID': int(label) + 1
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='utf-8-sig')
    
    def reduce_dimensions(self, method: str = 'tsne', n_components: int = 2) -> np.ndarray:
        """
        高维向量降维可视化
        ==================
        
        将高维文档向量降维到2D或3D空间，用于可视化聚类结果。
        
        算法原理 (t-SNE)
        ----------------
        t-SNE (t-distributed Stochastic Neighbor Embedding) 是一种非线性降维技术，
        特别适合高维数据的可视化。
        
        核心思想：
        1. **高维空间**: 使用高斯分布计算数据点之间的条件概率
           p(j|i) = exp(-||xᵢ-xⱼ||²/2σᵢ²) / Σₖ≠ᵢ exp(-||xᵢ-xₖ||²/2σᵢ²)
           
        2. **低维空间**: 使用t分布（自由度为1）计算相似度
           q(j|i) = (1+||yᵢ-yⱼ||²)⁻¹ / Σₖ≠ᵢ (1+||yᵢ-yₖ||²)⁻¹
           
        3. **优化目标**: 最小化两个分布之间的KL散度
           KL(P||Q) = Σᵢ Σⱼ pᵢⱼ log(pᵢⱼ/qᵢⱼ)
        
        为什么使用t分布？
        - t分布的重尾特性可以缓解"拥挤问题"
        - 允许不相似的点在低维空间中保持较远距离
        
        参数说明
        --------
        method : str, default='tsne'
            降维方法，目前支持 't-SNE'
            
        n_components : int, default=2
            目标维度，通常为2（2D可视化）或3（3D可视化）
        
        返回值
        ------
        np.ndarray
            降维后的坐标数组，形状为 (n_docs, n_components)
        
        注意事项
        --------
        - t-SNE是随机算法，每次运行结果可能略有不同
        - perplexity参数会根据数据量自动调整（范围5-30）
        - 降维前会进行标准化处理（StandardScaler）
        - t-SNE主要用于可视化，不适合作为特征提取方法
        
        示例
        ----
        >>> coords = clusterer.reduce_dimensions(method='tsne')
        >>> plt.scatter(coords[:, 0], coords[:, 1], c=clusterer.cluster_labels)
        """
        if len(self.doc_vectors) < 2:
            return self.doc_vectors
        
        # 标准化处理，确保各特征具有相同的尺度
        scaler = StandardScaler()
        scaled_vectors = scaler.fit_transform(self.doc_vectors)
        
        if method.lower() == 'tsne':
            # perplexity参数控制局部与全局结构的平衡
            # 通常设置为5-50，这里根据数据量自动调整
            perplexity = min(30, max(5, len(self.doc_vectors) - 1))
            
            tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
            return tsne.fit_transform(scaled_vectors)
        
        # 如果方法不支持，返回前n_components个特征
        return scaled_vectors[:, :n_components] if scaled_vectors.shape[1] >= n_components else scaled_vectors


class TextClassifier:
    """
    文本分类器 (Text Classifier)
    ============================
    
    提供半监督文本分类功能，支持用户定义分类标签、手动标注部分文档，
    然后基于已标注数据自动分类剩余文档。
    
    工作流程
    --------
    1. **定义标签**: 创建分类标签体系（如"正面"、"负面"、"中性"）
    2. **手动标注**: 为部分文档分配标签（建议每类至少标注5-10个样本）
    3. **训练模型**: 基于已标注数据训练KNN分类器
    4. **自动分类**: 使用训练好的模型预测未标注文档的类别
    
    核心算法 (KNN)
    --------------
    K-Nearest Neighbors (K近邻) 是一种基于实例的学习算法：
    
    算法原理：
    1. 对于待分类样本，计算它与所有训练样本的距离
    2. 找到距离最近的K个训练样本（邻居）
    3. 统计这K个邻居的类别分布
    4. 将待分类样本归为出现次数最多的类别（多数投票）
    
    数学表达：
    - 距离度量: d(x,y) = √(Σᵢ(xᵢ-yᵢ)²)  (欧氏距离)
    - 预测类别: ŷ = argmax_c Σᵢ∈Nₖ(x) I(yᵢ=c)
    
    算法特点：
    - 优点: 简单直观，无需训练过程，适合小规模数据
    - 缺点: 预测时计算量大，对特征尺度敏感，需要选择合适的K值
    
    属性说明
    --------
    doc_vectors : np.ndarray
        文档向量矩阵
    file_names : List[str]
        文档名称列表
    labels : Dict[str, str]
        已标注的文档标签映射 {文档名: 标签}
    available_labels : List[str]
        可用的分类标签列表
    classifier : KNeighborsClassifier
        训练好的KNN分类器
    predictions : Dict[str, str]
        自动分类的预测结果 {文档名: 预测标签}
    
    使用示例
    --------
    >>> # 创建分类器
    >>> classifier = TextClassifier(doc_vectors, file_names)
    >>> 
    >>> # 添加标签类别
    >>> classifier.add_label_category("正面评价")
    >>> classifier.add_label_category("负面评价")
    >>> 
    >>> # 手动标注
    >>> classifier.add_label("review1.txt", "正面评价")
    >>> classifier.add_label("review2.txt", "负面评价")
    >>> 
    >>> # 训练并预测
    >>> classifier.train_classifier(n_neighbors=3)
    >>> predictions = classifier.predict_unlabeled()
    
    Requirements: 3.5, 3.6
    """
    
    def __init__(self, doc_vectors: np.ndarray, file_names: List[str]):
        """
        初始化分类器
        
        参数说明
        --------
        doc_vectors : np.ndarray
            文档向量矩阵，形状为 (n_docs, n_features)
            - 建议使用与聚类相同的向量化方法
            
        file_names : List[str]
            文档名称列表，用于标识每个文档
        """
        self.doc_vectors = doc_vectors
        self.file_names = file_names
        self.labels: Dict[str, str] = {}  # doc_name -> label
        self.available_labels: List[str] = []  # 可用的分类标签
        self.classifier = None
        self.predictions: Dict[str, str] = {}  # 预测结果
    
    def add_label_category(self, label: str) -> bool:
        """
        添加分类标签类别
        
        在分类体系中添加一个新的标签类别。
        
        参数说明
        --------
        label : str
            标签名称，如"正面"、"负面"、"技术问题"等
        
        返回值
        ------
        bool
            是否添加成功
            - True: 标签添加成功
            - False: 标签为空或已存在
        
        示例
        ----
        >>> classifier.add_label_category("正面评价")
        True
        >>> classifier.add_label_category("正面评价")  # 重复添加
        False
        """
        if label and label not in self.available_labels:
            self.available_labels.append(label)
            return True
        return False
    
    def remove_label_category(self, label: str) -> bool:
        """
        移除分类标签类别
        
        从分类体系中移除一个标签类别，同时清除使用该标签的所有标注。
        
        参数说明
        --------
        label : str
            要移除的标签名称
        
        返回值
        ------
        bool
            是否移除成功
        
        注意事项
        --------
        移除标签会同时删除所有使用该标签的文档标注，请谨慎操作。
        """
        if label in self.available_labels:
            self.available_labels.remove(label)
            # 同时移除使用该标签的文档标注
            self.labels = {k: v for k, v in self.labels.items() if v != label}
            return True
        return False
    
    def add_label(self, doc_name: str, label: str) -> bool:
        """
        为文档添加分类标签
        
        手动为指定文档分配一个分类标签。
        
        参数说明
        --------
        doc_name : str
            文档名称，必须在file_names列表中
            
        label : str
            分类标签，必须在available_labels列表中
        
        返回值
        ------
        bool
            是否添加成功
            - True: 标注成功
            - False: 文档名或标签无效
        
        标注建议
        --------
        - 每个类别至少标注5-10个样本
        - 选择具有代表性的文档进行标注
        - 确保各类别的标注数量相对均衡
        """
        if doc_name in self.file_names and label in self.available_labels:
            self.labels[doc_name] = label
            return True
        return False
    
    def remove_label(self, doc_name: str) -> bool:
        """
        移除文档的分类标签
        
        取消对指定文档的标注。
        
        参数说明
        --------
        doc_name : str
            要取消标注的文档名称
        
        返回值
        ------
        bool
            是否移除成功
        """
        if doc_name in self.labels:
            del self.labels[doc_name]
            return True
        return False
    
    def get_labeled_documents(self) -> Dict[str, str]:
        """
        获取已标注的文档
        
        返回值
        ------
        Dict[str, str]
            文档名到标签的映射（副本，修改不影响原数据）
        """
        return self.labels.copy()
    
    def get_unlabeled_documents(self) -> List[str]:
        """
        获取未标注的文档
        
        返回值
        ------
        List[str]
            未标注的文档名称列表
        """
        return [name for name in self.file_names if name not in self.labels]
    
    def train_classifier(self, n_neighbors: int = 3) -> bool:
        """
        训练KNN分类器
        =============
        
        基于已标注数据训练K近邻分类器。
        
        算法说明
        --------
        KNN (K-Nearest Neighbors) 是一种懒惰学习算法：
        - 训练阶段仅存储训练数据，不构建显式模型
        - 预测阶段计算待分类样本与所有训练样本的距离
        - 选择距离最近的K个样本，通过多数投票决定类别
        
        参数说明
        --------
        n_neighbors : int, default=3
            K值，即考虑的最近邻居数量
            
            选择建议：
            - K值过小: 对噪声敏感，容易过拟合
            - K值过大: 决策边界过于平滑，可能欠拟合
            - 通常选择奇数以避免平票
            - 经验法则: K = √n，其中n是训练样本数
        
        返回值
        ------
        bool
            是否训练成功
            - True: 训练成功
            - False: 标注数据不足（少于2个或标签种类少于2）
        
        前置条件
        --------
        - 至少标注2个文档
        - 标注的文档至少包含2个不同的标签
        
        示例
        ----
        >>> success = classifier.train_classifier(n_neighbors=5)
        >>> if success:
        ...     predictions = classifier.predict_unlabeled()
        """
        if len(self.labels) < 2:
            return False
        
        # 准备训练数据
        train_indices = []
        train_labels = []
        
        for i, name in enumerate(self.file_names):
            if name in self.labels:
                train_indices.append(i)
                train_labels.append(self.labels[name])
        
        # 检查是否有至少2个不同的标签
        if len(set(train_labels)) < 2:
            return False
        
        X_train = self.doc_vectors[train_indices]
        
        # 训练KNN分类器
        # 确保K值不超过训练样本数
        n_neighbors = min(n_neighbors, len(train_indices))
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.classifier.fit(X_train, train_labels)
        
        return True
    
    def predict_unlabeled(self) -> Dict[str, str]:
        """
        预测未标注文档的分类
        ====================
        
        使用训练好的KNN分类器预测所有未标注文档的类别。
        
        算法流程
        --------
        对于每个未标注文档：
        1. 获取其文档向量
        2. 计算与所有训练样本的距离
        3. 找到K个最近邻
        4. 通过多数投票确定预测类别
        
        返回值
        ------
        Dict[str, str]
            文档名到预测标签的映射
            - 仅包含未标注的文档
            - 如果分类器未训练，返回空字典
        
        注意事项
        --------
        - 必须先调用train_classifier()训练模型
        - 预测结果保存在self.predictions中
        - 可通过get_all_labels()获取所有标签（包括手动和预测）
        
        示例
        ----
        >>> predictions = classifier.predict_unlabeled()
        >>> for doc, label in predictions.items():
        ...     print(f"{doc}: {label}")
        """
        if self.classifier is None:
            return {}
        
        self.predictions = {}
        
        for i, name in enumerate(self.file_names):
            if name not in self.labels:
                prediction = self.classifier.predict([self.doc_vectors[i]])[0]
                self.predictions[name] = prediction
        
        return self.predictions
    
    def get_all_labels(self) -> Dict[str, str]:
        """
        获取所有文档的标签
        
        合并手动标注和自动预测的结果。
        
        返回值
        ------
        Dict[str, str]
            文档名到标签的映射
            - 包括手动标注的文档
            - 包括自动分类的文档
            - 未分类的文档不在结果中
        """
        all_labels = self.labels.copy()
        all_labels.update(self.predictions)
        return all_labels
    
    def export_results(self) -> str:
        """
        导出分类结果为CSV格式
        
        将所有文档的分类结果导出为CSV格式字符串。
        
        返回值
        ------
        str
            CSV格式的分类结果字符串
            包含列：
            - 文档名: 文档名称
            - 分类标签: 分配的标签
            - 标注来源: "手动标注"、"自动分类"或"未分类"
        """
        all_labels = self.get_all_labels()
        
        data = []
        for name in self.file_names:
            label = all_labels.get(name, "未分类")
            source = "手动标注" if name in self.labels else ("自动分类" if name in self.predictions else "未分类")
            data.append({
                '文档名': name,
                '分类标签': label,
                '标注来源': source
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='utf-8-sig')


def create_doc_vectors(texts: List[List[str]], method: str = 'tfidf') -> np.ndarray:
    """
    文档向量化 - TF-IDF方法
    =======================
    
    将分词后的文本转换为数值向量，用于后续的聚类和分类分析。
    
    算法原理 (TF-IDF)
    -----------------
    TF-IDF (Term Frequency-Inverse Document Frequency) 是一种经典的文本特征提取方法，
    用于评估词语在文档集合中的重要程度。
    
    计算公式：
    
    1. **词频 (TF - Term Frequency)**
       TF(t,d) = 词t在文档d中出现的次数 / 文档d的总词数
       - 衡量词在单个文档中的重要性
       - 出现次数越多，TF值越高
    
    2. **逆文档频率 (IDF - Inverse Document Frequency)**
       IDF(t) = log(文档总数N / 包含词t的文档数df(t))
       - 衡量词的区分能力
       - 越少文档包含该词，IDF值越高
       - 常见词（如"的"、"是"）IDF值低
    
    3. **TF-IDF值**
       TF-IDF(t,d) = TF(t,d) × IDF(t)
       - 综合考虑词频和区分能力
       - 高TF-IDF值表示词在当前文档中重要且具有区分性
    
    参数说明
    --------
    texts : List[List[str]]
        分词后的文本列表
        - 外层列表: 每个元素对应一个文档
        - 内层列表: 该文档的词语序列
        - 示例: [["机器", "学习", "算法"], ["深度", "学习", "模型"]]
        
    method : str, default='tfidf'
        向量化方法，目前支持 'tfidf'
    
    返回值
    ------
    np.ndarray
        文档向量矩阵，形状为 (n_docs, n_features)
        - n_docs: 文档数量
        - n_features: 特征维度（最多1000个词）
        - 每行是一个文档的TF-IDF向量
    
    实现细节
    --------
    - 使用sklearn的TfidfVectorizer
    - max_features=1000: 限制词汇表大小，避免维度过高
    - 自动进行L2归一化
    
    使用建议
    --------
    - 输入文本应已完成分词和停用词过滤
    - 对于中文文本，确保分词质量
    - 如果文档数量很少，可能需要调整max_features
    
    示例
    ----
    >>> texts = [["机器", "学习"], ["深度", "学习", "神经网络"]]
    >>> vectors = create_doc_vectors(texts)
    >>> print(vectors.shape)  # (2, n_features)
    """
    if not texts:
        return np.array([])
    
    # 将词语列表转换为空格分隔的字符串
    # TfidfVectorizer需要字符串输入
    text_strings = [' '.join(words) for words in texts]
    
    if method == 'tfidf':
        # 创建TF-IDF向量化器
        # max_features=1000: 只保留最重要的1000个词，控制向量维度
        vectorizer = TfidfVectorizer(max_features=1000)
        
        # 拟合并转换文本
        # 返回稀疏矩阵，转换为密集数组
        doc_vectors = vectorizer.fit_transform(text_strings).toarray()
        return doc_vectors
    
    return np.array([])


def create_doc_vectors_from_topic_dist(doc_topic_dist: np.ndarray) -> np.ndarray:
    """
    基于主题分布的文档向量化
    ========================
    
    使用LDA主题模型的文档-主题分布作为文档向量。
    
    算法原理
    --------
    LDA (Latent Dirichlet Allocation) 主题模型将每个文档表示为主题的概率分布。
    这种表示方法具有以下特点：
    
    1. **语义表示**: 主题分布捕获了文档的语义信息
    2. **低维稠密**: 向量维度等于主题数（通常5-50），比TF-IDF更紧凑
    3. **概率解释**: 每个维度表示文档属于某主题的概率
    
    与TF-IDF的比较：
    - TF-IDF: 基于词频的稀疏高维表示
    - 主题分布: 基于语义的稠密低维表示
    
    参数说明
    --------
    doc_topic_dist : np.ndarray
        文档-主题分布矩阵，形状为 (n_docs, n_topics)
        - 由LDA模型训练得到
        - 每行是一个文档的主题分布
        - 每行元素之和为1（概率分布）
    
    返回值
    ------
    np.ndarray
        文档向量矩阵（直接返回输入的主题分布）
        - 如果输入为None，返回空数组
    
    使用场景
    --------
    - 已经训练了LDA主题模型
    - 希望基于主题语义进行聚类
    - 文档数量较少，TF-IDF效果不佳时
    
    示例
    ----
    >>> # 假设已有LDA模型的文档-主题分布
    >>> doc_topic_dist = lda_model.transform(corpus)
    >>> vectors = create_doc_vectors_from_topic_dist(doc_topic_dist)
    """
    if doc_topic_dist is None:
        return np.array([])
    return doc_topic_dist


def render_clustering_module():
    """
    渲染聚类与分类模块UI
    
    Requirements: 3.3, 3.7
    """
    st.subheader("🎯 文本聚类与分类")
    
    # 功能介绍与操作手册
    with st.expander("📖 功能介绍与操作手册", expanded=False):
        st.markdown("""
        ## 🎯 文本聚类与分类模块
        
        **功能概述**：提供文档自动聚类和分类功能，帮助研究者发现文档的内在分组结构。
        
        ---
        
        ### 🎯 使用场景
        
        | 场景 | 推荐方法 | 说明 |
        |------|----------|------|
        | 政策文档分类 | K-means + TF-IDF | 将政策文件按主题自动分组 |
        | 文献综述 | 层次聚类 | 发现研究文献的主题分布和层次结构 |
        | 内容组织 | 手动标注 + KNN | 为大量文档建立分类体系 |
        | 探索性分析 | 层次聚类 + 树状图 | 发现文档的自然分组 |
        
        ---
        
        ### 📊 文本聚类功能
        
        **K-means聚类**：
        - 将文档分成K个组，每组内文档相似度高
        - 适合大规模数据和已知聚类数量的场景
        - 需要预先指定聚类数量
        
        **层次聚类**：
        - 构建文档的层次结构，可视化为树状图
        - 适合探索性分析，无需预先指定聚类数
        - 可以在不同层次切割获得不同粒度的分组
        
        **聚类关键词**：
        - 自动提取每个聚类的代表性关键词
        - 帮助理解和命名聚类主题
        
        ---
        
        ### 🏷️ 文本分类功能
        
        **标签管理**：
        - 创建自定义分类标签体系
        - 支持添加、删除、修改标签
        
        **手动标注**：
        - 为部分文档手动分配标签作为训练样本
        - 建议每个类别至少标注5-10个样本
        
        **自动分类（KNN）**：
        - 基于K近邻算法自动分类未标注文档
        - 根据最相似的K个已标注文档投票决定类别
        
        ---
        
        ### 📋 操作步骤
        
        **文本聚类**：
        1. 选择向量化方法（TF-IDF或主题分布）
        2. 选择聚类算法（K-means或层次聚类）
        3. 设置聚类数量（K-means）或距离阈值（层次聚类）
        4. 点击"执行聚类"
        5. 查看聚类结果和可视化
        6. 导出聚类结果
        
        **文本分类**：
        1. 创建分类标签
        2. 手动标注部分文档
        3. 选择KNN的K值
        4. 点击"自动分类"
        5. 查看分类结果
        6. 导出分类结果
        
        ---
        
        ### ⚙️ 参数说明
        
        | 参数 | 范围 | 默认值 | 说明 |
        |------|------|--------|------|
        | 聚类数量(K) | 2-20 | 5 | K-means的聚类数 |
        | 距离阈值 | 0.1-2.0 | 1.0 | 层次聚类的切割阈值 |
        | KNN的K值 | 1-10 | 3 | 近邻数量 |
        
        ---
        
        ### 📈 可视化说明
        
        | 图表类型 | 说明 | 用途 |
        |----------|------|------|
        | 散点图 | t-SNE降维后的文档分布 | 直观展示聚类效果 |
        | 树状图 | 层次聚类的层次结构 | 分析文档的层次关系 |
        | 摘要表格 | 每个聚类的统计信息 | 了解聚类规模和特征 |
        
        ---
        
        ### 💡 使用建议
        
        **聚类数量选择**：
        - 可参考LDA模型的主题数
        - 使用轮廓系数评估不同K值的效果
        - 结合业务需求和可解释性选择
        
        **向量化方法选择**：
        - TF-IDF：通用场景，不依赖LDA模型
        - 主题分布：已训练LDA模型时推荐使用
        
        **分类标注建议**：
        - 每个类别至少标注5-10个样本
        - 选择典型、清晰的文档进行标注
        - 标注样本应覆盖类别的多样性
        
        ---
        
        ### ❓ 常见问题
        
        **Q: 如何选择合适的聚类数量？**
        A: 可以尝试多个K值，比较轮廓系数，选择效果最好的。也可参考LDA主题数。
        
        **Q: 聚类结果不理想怎么办？**
        A: 尝试调整预处理参数、更换向量化方法、或调整聚类数量。
        
        **Q: KNN分类准确率低怎么办？**
        A: 增加标注样本数量，或调整K值。确保标注样本具有代表性。
        """)
    
    # 检查数据
    if not st.session_state.get("texts"):
        st.warning("⚠️ 请先完成文本预处理")
        return
    
    texts = st.session_state.get("texts", [])
    file_names = st.session_state.get("file_names", [])
    doc_topic_dist = st.session_state.get("doc_topic_dist")
    
    if len(texts) < 2:
        st.warning("⚠️ 至少需要2个文档才能进行聚类分析")
        return
    
    # 创建子标签页
    cluster_tabs = st.tabs(["📊 文本聚类", "🏷️ 文本分类"])
    
    # ========== 文本聚类标签页 ==========
    with cluster_tabs[0]:
        render_clustering_tab(texts, file_names, doc_topic_dist)
    
    # ========== 文本分类标签页 ==========
    with cluster_tabs[1]:
        render_classification_tab(texts, file_names, doc_topic_dist)


def render_clustering_tab(texts: List[List[str]], file_names: List[str], doc_topic_dist: Optional[np.ndarray]):
    """渲染聚类标签页"""
    st.markdown("### 文档聚类分析")
    
    # 算法说明折叠面板
    with st.expander("📚 算法说明与适用场景", expanded=False):
        st.markdown("""
        #### 🔷 文档向量化方法
        
        | 方法 | 原理 | 适用场景 | 优缺点 |
        |------|------|----------|--------|
        | **TF-IDF** | 基于词频-逆文档频率计算词语权重 | 通用文本聚类、关键词提取 | ✅ 简单高效，不依赖预训练模型<br>❌ 忽略词序和语义 |
        | **主题分布** | 使用LDA模型的文档-主题分布向量 | 基于主题的文档分组 | ✅ 捕捉文档主题语义<br>❌ 需要先训练LDA模型 |
        
        ---
        
        #### 🔷 聚类算法
        
        | 算法 | 原理 | 适用场景 | 优缺点 |
        |------|------|----------|--------|
        | **K-means** | 迭代优化，最小化簇内距离平方和 | 大规模数据、球形分布的聚类 | ✅ 速度快，适合大数据<br>❌ 需预设K值，对异常值敏感 |
        | **层次聚类** | 自底向上合并相似文档构建层次结构 | 探索性分析、需要层次关系 | ✅ 可视化树状图，无需预设K<br>❌ 计算复杂度高 |
        
        ---
        
        #### 🔷 层次聚类链接方法
        
        | 方法 | 计算方式 | 适用场景 |
        |------|----------|----------|
        | **ward** | 最小化合并后的方差增量 | 🌟 推荐默认选项，产生紧凑均匀的聚类 |
        | **complete** | 使用两簇间最大距离 | 适合发现紧凑的球形聚类 |
        | **average** | 使用两簇间平均距离 | 对异常值较鲁棒 |
        | **single** | 使用两簇间最小距离 | 适合发现链状或不规则形状聚类 |
        
        ---
        
        #### 💡 使用建议
        
        1. **探索性分析**：先用层次聚类观察树状图，确定合适的聚类数量
        2. **大规模数据**：优先使用K-means，效率更高
        3. **主题相关分析**：如果已训练LDA模型，建议使用主题分布向量
        4. **聚类数量选择**：一般从3-5个开始尝试，观察聚类质量
        """)
    
    st.markdown("---")
    
    # 参数设置区域
    st.markdown("#### ⚙️ 参数设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 选择向量化方法
        vector_options = ["TF-IDF", "主题分布"] if doc_topic_dist is not None else ["TF-IDF"]
        vector_method = st.selectbox(
            "📊 文档向量化方法",
            vector_options,
            key="cluster_vector_method",
            help="选择如何将文档转换为数值向量"
        )
        
        # 向量化方法说明
        if vector_method == "TF-IDF":
            st.info("💡 **TF-IDF**：基于词频统计，适合通用文本聚类分析")
        else:
            st.info("💡 **主题分布**：基于LDA主题模型，适合按主题内容分组")
    
    with col2:
        # 选择聚类算法
        algorithm = st.selectbox(
            "🔬 聚类算法",
            ["K-means", "层次聚类"],
            key="cluster_algorithm_select",
            help="选择聚类算法"
        )
        
        # 算法说明
        if algorithm == "K-means":
            st.info("💡 **K-means**：快速高效，适合大规模数据和球形聚类")
        else:
            st.info("💡 **层次聚类**：可生成树状图，适合探索文档层次关系")
    
    # 第二行参数
    col3, col4 = st.columns(2)
    
    with col3:
        # 聚类数量
        max_clusters = min(10, len(texts))
        n_clusters = st.slider(
            "📌 聚类数量 (K)",
            min_value=2,
            max_value=max_clusters,
            value=min(3, max_clusters),
            key="cluster_n_clusters",
            help="将文档分成多少个组"
        )
        st.caption(f"当前文档数：{len(texts)}，建议聚类数：2-{min(5, max_clusters)}")
    
    with col4:
        # 层次聚类的额外参数
        if algorithm == "层次聚类":
            linkage_method = st.selectbox(
                "🔗 链接方法",
                ["ward", "complete", "average", "single"],
                key="cluster_linkage_method",
                help="层次聚类的距离计算方式"
            )
            
            # 链接方法说明
            linkage_descriptions = {
                "ward": "🌟 推荐 - 最小化方差，产生紧凑均匀的聚类",
                "complete": "使用最大距离，适合紧凑球形聚类",
                "average": "使用平均距离，对异常值较鲁棒",
                "single": "使用最小距离，适合链状聚类"
            }
            st.caption(linkage_descriptions.get(linkage_method, ""))
        else:
            linkage_method = "ward"
            st.empty()  # 占位
    
    st.markdown("---")
    
    # 执行聚类按钮
    if st.button("🚀 执行聚类", key="run_clustering", type="primary"):
        with st.spinner("正在执行聚类分析..."):
            try:
                # 创建文档向量
                if vector_method == "主题分布" and doc_topic_dist is not None:
                    doc_vectors = create_doc_vectors_from_topic_dist(doc_topic_dist)
                else:
                    doc_vectors = create_doc_vectors(texts)
                
                if len(doc_vectors) == 0:
                    st.error("无法创建文档向量")
                    return
                
                # 创建聚类器
                clusterer = TextClusterer(doc_vectors, file_names)
                
                # 执行聚类
                if algorithm == "K-means":
                    cluster_labels = clusterer.kmeans_clustering(n_clusters)
                else:
                    cluster_labels = clusterer.hierarchical_clustering(n_clusters, linkage_method)
                
                # 保存到会话状态
                st.session_state["cluster_labels"] = cluster_labels
                st.session_state["clusterer"] = clusterer
                st.session_state["cluster_keywords"] = clusterer.get_all_cluster_keywords(texts)
                st.session_state["cluster_algorithm"] = algorithm
                st.session_state["cluster_vector_method_used"] = vector_method
                
                log_message(f"聚类完成：{algorithm}，{n_clusters}个聚类", level="success")
                st.success(f"✅ 聚类完成！使用 {algorithm} 算法将 {len(texts)} 个文档分为 {n_clusters} 个聚类")
                
            except Exception as e:
                st.error(f"聚类失败：{str(e)}")
                log_message(f"聚类失败：{str(e)}", level="error")
    
    # 显示聚类结果
    if st.session_state.get("clusterer") is not None:
        clusterer = st.session_state["clusterer"]
        used_algorithm = st.session_state.get("cluster_algorithm", "未知")
        used_vector = st.session_state.get("cluster_vector_method_used", "未知")
        
        st.markdown("---")
        st.markdown("### 📋 聚类结果")
        
        # 显示当前使用的配置
        st.caption(f"📊 向量化方法：{used_vector} | 🔬 聚类算法：{used_algorithm} | 📌 聚类数：{clusterer.n_clusters}")
        
        # 聚类摘要
        summary_df = clusterer.get_cluster_summary(texts)
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True)
        
        # 可视化选项
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown("#### 📊 散点图")
            try:
                # 降维可视化
                coords = clusterer.reduce_dimensions(method='tsne')
                
                viz_df = pd.DataFrame({
                    'x': coords[:, 0],
                    'y': coords[:, 1],
                    '文档': file_names,
                    '聚类': [f'聚类{i+1}' for i in clusterer.cluster_labels]
                })
                
                fig = px.scatter(
                    viz_df,
                    x='x', y='y',
                    color='聚类',
                    hover_data=['文档'],
                    title='文档聚类散点图 (t-SNE降维)'
                )
                fig.update_layout(
                    xaxis_title='',
                    yaxis_title='',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"无法生成散点图：{str(e)}")
        
        with viz_col2:
            st.markdown("#### 🌳 树状图")
            if clusterer.linkage_matrix is not None:
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    dendrogram(
                        clusterer.linkage_matrix,
                        labels=file_names,
                        leaf_rotation=90,
                        leaf_font_size=8,
                        ax=ax
                    )
                    ax.set_title('层次聚类树状图')
                    ax.set_xlabel('文档')
                    ax.set_ylabel('距离')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.warning(f"无法生成树状图：{str(e)}")
            else:
                st.info("树状图仅在使用层次聚类时可用")
        
        # 聚类关键词详情
        st.markdown("#### 🔑 聚类关键词")
        keywords = st.session_state.get("cluster_keywords", {})
        if keywords:
            keyword_cols = st.columns(min(3, len(keywords)))
            for i, (cluster_id, words) in enumerate(keywords.items()):
                with keyword_cols[i % len(keyword_cols)]:
                    st.markdown(f"**聚类 {cluster_id + 1}**")
                    st.write(", ".join(words[:10]))
        
        # 导出结果
        st.markdown("---")
        if st.button("💾 导出聚类结果", key="export_clustering"):
            csv_content = clusterer.export_results()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                label="📥 下载CSV",
                data=csv_content.encode('utf-8-sig'),
                file_name=f"clustering_results_{timestamp}.csv",
                mime="text/csv"
            )


def render_classification_tab(texts: List[List[str]], file_names: List[str], doc_topic_dist: Optional[np.ndarray]):
    """渲染分类标签页"""
    st.markdown("### 文档分类")
    
    # 算法说明折叠面板
    with st.expander("📚 分类算法说明与适用场景", expanded=False):
        st.markdown("""
        #### 🔷 分类方法：K近邻 (KNN)
        
        **原理**：根据文档在向量空间中的位置，找到K个最相似的已标注文档，通过投票决定新文档的类别。
        
        | 特点 | 说明 |
        |------|------|
        | **优点** | 简单直观，无需复杂训练，适合小规模数据 |
        | **缺点** | 对标注样本质量敏感，需要足够的标注数据 |
        | **适用场景** | 文档数量适中，有部分已知类别的文档 |
        
        ---
        
        #### 💡 使用流程
        
        1. **创建标签**：定义分类体系（如：政策类、经济类、科技类）
        2. **手动标注**：为部分文档（建议每类至少2-3个）分配标签
        3. **自动分类**：系统基于已标注样本，自动为剩余文档分类
        
        ---
        
        #### ⚠️ 注意事项
        
        - 至少需要标注 **2个不同类别** 的文档才能训练分类器
        - 标注样本应具有 **代表性**，覆盖各类别的典型文档
        - 标注数量越多，分类准确率越高
        - 建议每个类别至少标注 **3-5个** 样本
        """)
    
    st.markdown("---")
    
    # 初始化分类器
    if "text_classifier" not in st.session_state or st.session_state.get("classifier_file_names") != file_names:
        # 创建文档向量
        if doc_topic_dist is not None:
            doc_vectors = create_doc_vectors_from_topic_dist(doc_topic_dist)
        else:
            doc_vectors = create_doc_vectors(texts)
        
        if len(doc_vectors) > 0:
            st.session_state["text_classifier"] = TextClassifier(doc_vectors, file_names)
            st.session_state["classifier_file_names"] = file_names
    
    classifier = st.session_state.get("text_classifier")
    if classifier is None:
        st.warning("⚠️ 无法初始化分类器")
        return
    
    # 标签管理
    st.markdown("#### 1️⃣ 标签管理")
    st.caption("💡 创建分类标签体系，如：政策类、经济类、科技类等")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        new_label = st.text_input("添加新标签", key="new_classification_label", placeholder="输入标签名称，如：政策类")
    with col2:
        st.write("")  # 占位
        st.write("")  # 占位
        if st.button("➕ 添加", key="add_label_btn"):
            if new_label:
                if classifier.add_label_category(new_label):
                    st.success(f"已添加标签：{new_label}")
                    st.rerun()
                else:
                    st.warning("标签已存在或无效")
    
    # 显示现有标签
    if classifier.available_labels:
        st.write("**现有标签：**", ", ".join(classifier.available_labels))
        
        # 删除标签
        label_to_remove = st.selectbox(
            "选择要删除的标签",
            [""] + classifier.available_labels,
            key="label_to_remove"
        )
        if label_to_remove and st.button("🗑️ 删除标签", key="remove_label_btn"):
            if classifier.remove_label_category(label_to_remove):
                st.success(f"已删除标签：{label_to_remove}")
                st.rerun()
    else:
        st.info("请先添加分类标签")
    
    st.markdown("---")
    
    # 手动标注
    st.markdown("#### 2️⃣ 手动标注文档")
    st.caption("💡 为部分文档分配标签作为训练样本，建议每个类别至少标注3-5个文档")
    
    if classifier.available_labels:
        # 显示未标注的文档
        unlabeled_docs = classifier.get_unlabeled_documents()
        labeled_docs = classifier.get_labeled_documents()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**未标注文档**")
            if unlabeled_docs:
                selected_doc = st.selectbox(
                    "选择文档",
                    unlabeled_docs,
                    key="doc_to_label"
                )
                selected_label = st.selectbox(
                    "选择标签",
                    classifier.available_labels,
                    key="label_for_doc"
                )
                if st.button("✅ 标注", key="label_doc_btn"):
                    if classifier.add_label(selected_doc, selected_label):
                        st.success(f"已标注：{selected_doc} → {selected_label}")
                        st.rerun()
            else:
                st.info("所有文档已标注")
        
        with col2:
            st.markdown("**已标注文档**")
            if labeled_docs:
                labeled_df = pd.DataFrame([
                    {"文档": k, "标签": v} for k, v in labeled_docs.items()
                ])
                st.dataframe(labeled_df, use_container_width=True, height=200)
                
                # 移除标注
                doc_to_unlabel = st.selectbox(
                    "选择要移除标注的文档",
                    [""] + list(labeled_docs.keys()),
                    key="doc_to_unlabel"
                )
                if doc_to_unlabel and st.button("🗑️ 移除标注", key="unlabel_doc_btn"):
                    if classifier.remove_label(doc_to_unlabel):
                        st.success(f"已移除标注：{doc_to_unlabel}")
                        st.rerun()
            else:
                st.info("暂无已标注文档")
        
        st.markdown("---")
        
        # 自动分类
        st.markdown("#### 3️⃣ 自动分类")
        st.caption("💡 基于KNN算法，利用已标注文档自动为未标注文档分配类别")
        
        labeled_count = len(labeled_docs)
        unlabeled_count = len(unlabeled_docs)
        
        # 显示标注统计
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("已标注", f"{labeled_count} 个")
        with col_stat2:
            st.metric("未标注", f"{unlabeled_count} 个")
        with col_stat3:
            st.metric("标签类别", f"{len(classifier.available_labels)} 个")
        
        if labeled_count >= 2 and unlabeled_count > 0:
            # 检查是否有至少2个不同的标签
            unique_labels = set(labeled_docs.values())
            if len(unique_labels) >= 2:
                st.success(f"✅ 已满足训练条件：{len(unique_labels)} 个不同类别，{labeled_count} 个标注样本")
            else:
                st.warning(f"⚠️ 当前只有 {len(unique_labels)} 个类别，需要至少2个不同类别才能训练")
            
            if st.button("🤖 自动分类未标注文档", key="auto_classify_btn", type="primary"):
                with st.spinner("正在训练分类器并预测..."):
                    if classifier.train_classifier():
                        predictions = classifier.predict_unlabeled()
                        st.session_state["classification_predictions"] = predictions
                        st.success(f"✅ 已自动分类 {len(predictions)} 个文档")
                        log_message(f"自动分类完成：{len(predictions)}个文档", level="success")
                    else:
                        st.error("训练分类器失败，请确保至少有2个不同标签的文档")
        elif labeled_count < 2:
            st.info("📌 请至少标注2个文档（且包含不同标签）才能进行自动分类")
        else:
            st.success("🎉 所有文档已标注，无需自动分类")
        
        # 显示分类结果
        all_labels = classifier.get_all_labels()
        if all_labels:
            st.markdown("---")
            st.markdown("#### 📋 分类结果")
            
            result_data = []
            for name in file_names:
                label = all_labels.get(name, "未分类")
                source = "手动" if name in labeled_docs else ("自动" if name in classifier.predictions else "未分类")
                result_data.append({
                    "文档": name,
                    "标签": label,
                    "来源": source
                })
            
            result_df = pd.DataFrame(result_data)
            st.dataframe(result_df, use_container_width=True)
            
            # 导出结果
            if st.button("💾 导出分类结果", key="export_classification"):
                csv_content = classifier.export_results()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                st.download_button(
                    label="📥 下载CSV",
                    data=csv_content.encode('utf-8-sig'),
                    file_name=f"classification_results_{timestamp}.csv",
                    mime="text/csv"
                )
    else:
        st.info("请先在上方添加分类标签")
