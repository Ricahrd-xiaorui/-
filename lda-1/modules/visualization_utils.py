import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
import matplotlib.cm as cm
import streamlit as st

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    st.warning("未能设置中文字体，可能导致中文显示异常")

def calculate_topic_distribution(doc_topic_dist):
    """
    计算主题分布
    
    参数:
        doc_topic_dist: 文档-主题分布矩阵
        
    返回:
        pandas DataFrame: 包含主题分布信息的数据框
    """
    # 获取每个文档的主要主题
    dominant_topics = np.argmax(doc_topic_dist, axis=1)
    
    # 计算每个主题的文档数量
    topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
    
    # 转换为DataFrame
    topic_dist = pd.DataFrame({
        '主题ID': topic_counts.index,
        '文档数量': topic_counts.values,
        '文档比例': topic_counts.values / len(doc_topic_dist)
    })
    
    return topic_dist

def plot_topic_distribution(topic_dist, chart_type='条形图', color_scheme='viridis'):
    """
    绘制主题分布图
    
    参数:
        topic_dist: 主题分布数据框
        chart_type: 图表类型 ('条形图', '饼图', '面积图')
        color_scheme: 颜色方案
        
    返回:
        matplotlib figure: 绘制的图表
    """
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 设置颜色映射
    cmap = plt.get_cmap(color_scheme)
    colors = cmap(np.linspace(0, 1, len(topic_dist)))
    
    # 根据图表类型绘制
    if chart_type == '条形图':
        bars = ax.bar(topic_dist['主题ID'] + 1, topic_dist['文档数量'], color=colors)
        
        # 在柱形上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.0f}', ha='center', va='bottom')
        
        ax.set_xlabel('主题ID')
        ax.set_ylabel('文档数量')
        ax.set_title('主题分布')
        ax.set_xticks(topic_dist['主题ID'] + 1)
        
    elif chart_type == '饼图':
        wedges, texts, autotexts = ax.pie(
            topic_dist['文档数量'], 
            labels=[f'主题 {i+1}' for i in topic_dist['主题ID']], 
            autopct='%1.1f%%',
            colors=colors,
            shadow=True,
            startangle=90
        )
        ax.axis('equal')  # 确保饼图是圆的
        ax.set_title('主题分布')
        
        # 设置文本属性
        plt.setp(autotexts, size=10, weight='bold')
        
    elif chart_type == '面积图':
        # 创建堆叠面积图数据
        x = np.arange(len(topic_dist))
        y = topic_dist['文档数量']
        
        # 绘制面积图
        ax.fill_between(x, y, color=colors[0], alpha=0.7)
        ax.plot(x, y, 'o-', color=colors[-1], linewidth=2)
        
        # 设置标签
        ax.set_xlabel('主题序号')
        ax.set_ylabel('文档数量')
        ax.set_title('主题分布面积图')
        ax.set_xticks(x)
        ax.set_xticklabels([f'主题 {i+1}' for i in topic_dist['主题ID']])
        
        # 添加数据标签
        for i, value in enumerate(y):
            ax.text(i, value + 0.5, f'{value:.0f}', ha='center')
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 设置图表样式
    plt.tight_layout()
    
    return fig

def get_topic_term_dists(lda_model, dictionary):
    """
    获取主题-词分布
    
    参数:
        lda_model: LDA模型
        dictionary: 词典
        
    返回:
        numpy array: 主题-词分布矩阵
    """
    # 获取主题-词分布
    topic_term_dists = np.zeros((lda_model.num_topics, len(dictionary)))
    
    for topic_id in range(lda_model.num_topics):
        for word_id, weight in lda_model.get_topic_terms(topic_id, topn=len(dictionary)):
            topic_term_dists[topic_id, word_id] = weight
    
    return topic_term_dists

def plot_topic_term_heatmap(topic_term_dists, dictionary, top_n_topics=10, top_n_words=10, normalize=True, cmap='viridis'):
    """
    绘制主题-词热力图
    
    参数:
        topic_term_dists: 主题-词分布矩阵
        dictionary: 词典
        top_n_topics: 显示的主题数量
        top_n_words: 每个主题显示的词数量
        normalize: 是否归一化词权重
        cmap: 颜色方案
        
    返回:
        matplotlib figure: 热力图
    """
    # 限制主题数量
    if top_n_topics > topic_term_dists.shape[0]:
        top_n_topics = topic_term_dists.shape[0]
    
    # 获取每个主题的前N个词
    top_words_idx = np.zeros((top_n_topics, top_n_words), dtype=int)
    top_words_val = np.zeros((top_n_topics, top_n_words))
    
    for topic_idx in range(top_n_topics):
        # 获取词索引按权重排序
        sorted_word_idx = np.argsort(topic_term_dists[topic_idx])[::-1]
        top_words_idx[topic_idx] = sorted_word_idx[:top_n_words]
        top_words_val[topic_idx] = topic_term_dists[topic_idx, sorted_word_idx[:top_n_words]]
    
    # 获取对应的词
    words = []
    for topic_idx in range(top_n_topics):
        words.append([dictionary[idx] for idx in top_words_idx[topic_idx]])
    
    # 创建热力图数据
    heatmap_data = np.zeros((top_n_topics, top_n_words))
    for topic_idx in range(top_n_topics):
        heatmap_data[topic_idx] = top_words_val[topic_idx]
    
    # 归一化
    if normalize:
        for topic_idx in range(top_n_topics):
            max_val = np.max(heatmap_data[topic_idx])
            if max_val > 0:
                heatmap_data[topic_idx] = heatmap_data[topic_idx] / max_val
    
    # 创建标签
    x_labels = []
    for topic_idx in range(top_n_topics):
        for word_idx in range(top_n_words):
            if len(words[topic_idx][word_idx]) > 15:
                words[topic_idx][word_idx] = words[topic_idx][word_idx][:12] + '...'
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建热力图
    sns.heatmap(
        heatmap_data, 
        annot=False, 
        cmap=cmap, 
        ax=ax, 
        linewidths=0.5, 
        linecolor='white',
        cbar_kws={'label': '归一化词权重' if normalize else '词权重'}
    )
    
    # 设置标签
    ax.set_yticks(np.arange(top_n_topics) + 0.5)
    ax.set_yticklabels([f'主题 {i+1}' for i in range(top_n_topics)])
    
    # 设置x轴标签为每个主题的前N个词
    all_x_ticks = []
    all_x_labels = []
    
    for topic_idx in range(top_n_topics):
        for word_idx in range(top_n_words):
            all_x_ticks.append(topic_idx * top_n_words + word_idx + 0.5)
            all_x_labels.append(words[topic_idx][word_idx])
    
    # 设置标题和标签
    ax.set_title('主题-词热力图', fontsize=16)
    ax.set_xlabel('关键词', fontsize=12)
    ax.set_ylabel('主题', fontsize=12)
    
    # 添加网格线
    ax.grid(False)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def generate_wordcloud_for_topic(lda_model, dictionary, topic_id, max_words=50, colormap='viridis'):
    """
    为指定主题生成词云
    
    参数:
        lda_model: LDA模型
        dictionary: 词典
        topic_id: 主题ID
        max_words: 最大显示词数
        colormap: 颜色方案
        
    返回:
        matplotlib figure: 词云图
    """
    # 获取主题的词分布
    topic_words = lda_model.get_topic_terms(topic_id, topn=100)
    
    # 创建词-权重字典
    word_weights = {dictionary[word_id]: weight for word_id, weight in topic_words}
    
    # 创建词云
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=max_words,
        colormap=colormap,
        contour_width=1,
        contour_color='steelblue',
        prefer_horizontal=0.9,
        font_path='simhei.ttf' if 'simhei.ttf' in plt.rcParams['font.sans-serif'] else None
    )
    
    # 生成词云
    wordcloud.generate_from_frequencies(word_weights)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'主题 {topic_id+1} 词云', fontsize=16)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def plot_document_clustering(doc_topic_dist, method='t-SNE', clustering_method='基于主题', cmap='viridis', marker_size=40):
    """
    绘制文档聚类图
    
    参数:
        doc_topic_dist: 文档-主题分布矩阵
        method: 降维方法 ('t-SNE', 'PCA', 'UMAP')
        clustering_method: 聚类方法 ('基于主题', 'K-Means', 'DBSCAN', '层次聚类')
        cmap: 颜色方案
        marker_size: 散点大小
        
    返回:
        matplotlib figure: 聚类图
    """
    # 降维
    if method == 't-SNE':
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(doc_topic_dist)-1))
        doc_tsne = tsne.fit_transform(doc_topic_dist)
        embedding = doc_tsne
        
    elif method == 'PCA':
        # PCA降维
        pca = PCA(n_components=2, random_state=42)
        doc_pca = pca.fit_transform(doc_topic_dist)
        embedding = doc_pca
        
    elif method == 'UMAP':
        # 如果UMAP可用
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            doc_umap = reducer.fit_transform(doc_topic_dist)
            embedding = doc_umap
        except ImportError:
            st.warning("UMAP未安装，使用t-SNE替代")
            tsne = TSNE(n_components=2, random_state=42)
            embedding = tsne.fit_transform(doc_topic_dist)
    
    # 聚类
    if clustering_method == '基于主题':
        # 使用主题分布中的最大值作为类别
        labels = np.argmax(doc_topic_dist, axis=1)
        n_clusters = len(np.unique(labels))
        
    elif clustering_method == 'K-Means':
        # 使用K-Means聚类
        n_clusters = min(8, len(doc_topic_dist))  # 最多8个类别
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(doc_topic_dist)
        
    elif clustering_method == 'DBSCAN':
        # 使用DBSCAN聚类
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        labels = dbscan.fit_predict(doc_topic_dist)
        n_clusters = len(np.unique(labels))
        
    elif clustering_method == '层次聚类':
        # 使用层次聚类
        dist_matrix = pdist(doc_topic_dist, metric='euclidean')
        linkage_matrix = sch.linkage(dist_matrix, method='ward')
        n_clusters = min(8, len(doc_topic_dist))  # 最多8个类别
        labels = sch.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        labels = labels - 1  # 从0开始
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 设置颜色映射
    cmap_obj = plt.get_cmap(cmap)
    
    # 为每个类别绘制散点
    for i in range(max(n_clusters, 1)):
        idx = labels == i
        if np.any(idx):  # 确保该类别有数据点
            ax.scatter(
                embedding[idx, 0], 
                embedding[idx, 1], 
                s=marker_size, 
                c=[cmap_obj(i / max(n_clusters, 1))], 
                label=f'类别 {i+1}',
                alpha=0.7,
                edgecolors='w',
                linewidth=0.5
            )
    
    # 设置图表属性
    ax.set_title(f'文档聚类 ({method} + {clustering_method})', fontsize=15)
    ax.set_xlabel(f'{method} 维度 1', fontsize=12)
    ax.set_ylabel(f'{method} 维度 2', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title="聚类", fontsize=10)
    
    # 添加轮廓系数
    if len(np.unique(labels)) > 1 and len(labels) > 1:
        try:
            silhouette_avg = silhouette_score(doc_topic_dist, labels)
            ax.text(0.02, 0.02, f'轮廓系数: {silhouette_avg:.3f}', 
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(facecolor='white', alpha=0.8))
        except:
            pass
    
    # 调整布局
    plt.tight_layout()
    
    return fig 