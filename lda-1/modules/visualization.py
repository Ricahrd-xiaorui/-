import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud

def calculate_topic_distribution(doc_topic_dist):
    """计算主题分布统计信息"""
    # 获取每个文档的主要主题
    dominant_topics = np.argmax(doc_topic_dist, axis=1)
    
    # 计算每个主题的文档数量
    topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
    
    # 创建数据框
    topic_dist = pd.DataFrame({
        '主题': [f'主题 {i+1}' for i in topic_counts.index],
        '文档数量': topic_counts.values,
        '文档比例': topic_counts.values / len(doc_topic_dist)
    })
    
    return topic_dist

def plot_topic_distribution(topic_dist, chart_type="条形图", color_scheme="viridis"):
    """绘制主题分布图表"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if chart_type == "条形图":
        # 绘制条形图
        sns.barplot(x='主题', y='文档数量', data=topic_dist, palette=color_scheme, ax=ax)
        ax.set_title('主题分布')
        ax.set_ylabel('文档数量')
        
        # 添加数值标签
        for i, v in enumerate(topic_dist['文档数量']):
            ax.text(i, v + 0.5, str(v), ha='center')
            
    elif chart_type == "饼图":
        # 绘制饼图
        ax.pie(
            topic_dist['文档数量'], 
            labels=topic_dist['主题'],
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
            colors=plt.cm.get_cmap(color_scheme)(np.linspace(0, 1, len(topic_dist)))
        )
        ax.axis('equal')
        ax.set_title('主题分布')
        
    elif chart_type == "面积图":
        # 创建面积图数据
        x = range(len(topic_dist))
        y = topic_dist['文档数量']
        
        # 绘制面积图
        ax.fill_between(x, y, color=plt.cm.get_cmap(color_scheme)(0.5), alpha=0.6)
        ax.plot(x, y, 'o-', color=plt.cm.get_cmap(color_scheme)(0.8))
        
        # 设置x轴标签
        ax.set_xticks(x)
        ax.set_xticklabels(topic_dist['主题'])
        ax.set_title('主题分布')
        ax.set_ylabel('文档数量')
        
        # 添加数值标签
        for i, v in enumerate(y):
            ax.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    return fig

def get_topic_term_dists(lda_model, dictionary):
    """获取主题-词分布"""
    topic_term_dists = np.zeros((lda_model.num_topics, len(dictionary)))
    
    for topic_id in range(lda_model.num_topics):
        topic_dist = lda_model.get_topic_terms(topic_id, topn=len(dictionary))
        for term_id, prob in topic_dist:
            topic_term_dists[topic_id, term_id] = prob
    
    return topic_term_dists

def plot_topic_term_heatmap(topic_term_dists, dictionary, top_n_topics=None, top_n_words=10, normalize=True, cmap="YlGnBu"):
    """绘制主题-词热力图"""
    # 确定要显示的主题数量
    if top_n_topics is None or top_n_topics > topic_term_dists.shape[0]:
        top_n_topics = topic_term_dists.shape[0]
    
    # 对每个主题，获取概率最高的词
    top_words_indices = []
    for topic_idx in range(top_n_topics):
        # 获取该主题下词的概率排序
        topic_dist = topic_term_dists[topic_idx]
        top_indices = np.argsort(topic_dist)[-top_n_words:][::-1]
        top_words_indices.extend(top_indices)
    
    # 去重
    top_words_indices = list(set(top_words_indices))
    
    # 创建热力图数据
    heatmap_data = topic_term_dists[:top_n_topics, top_words_indices]
    
    # 归一化处理
    if normalize:
        heatmap_data = heatmap_data / np.max(heatmap_data, axis=1, keepdims=True)
    
    # 获取词语
    words = [dictionary[idx] for idx in top_words_indices]
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        annot=False,
        cmap=cmap,
        xticklabels=words,
        yticklabels=[f'主题 {i+1}' for i in range(top_n_topics)],
        ax=ax
    )
    
    # 设置标题和标签
    ax.set_title('主题-词分布热力图')
    ax.set_xlabel('词语')
    ax.set_ylabel('主题')
    
    # 旋转x轴标签以提高可读性
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def generate_wordcloud_for_topic(lda_model, dictionary, topic_id, max_words=50, colormap="viridis"):
    """为指定主题生成词云"""
    # 获取主题词分布
    topic_words = lda_model.get_topic_terms(topic_id, topn=max_words)
    word_dict = {dictionary[id]: value for id, value in topic_words}
    
    # 创建词云
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=colormap,
        max_words=max_words,
        random_state=42
    ).generate_from_frequencies(word_dict)
    
    # 绘制词云
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'主题 {topic_id+1} 词云')
    
    return fig

def plot_document_clustering(doc_topic_dist, method='t-SNE', clustering_method='基于主题', cmap='viridis', marker_size=40):
    """绘制文档聚类图"""
    # 降维
    if method == 't-SNE':
        from sklearn.manifold import TSNE
        embedding = TSNE(n_components=2, random_state=42).fit_transform(doc_topic_dist)
    elif method == 'PCA':
        from sklearn.decomposition import PCA
        embedding = PCA(n_components=2, random_state=42).fit_transform(doc_topic_dist)
    elif method == 'UMAP':
        try:
            import umap
            embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(doc_topic_dist)
        except ImportError:
            st.error("请安装UMAP库: pip install umap-learn")
            return None
    
    # 聚类
    if clustering_method == '基于主题':
        # 使用文档的主要主题作为聚类标签
        labels = np.argmax(doc_topic_dist, axis=1)
    elif clustering_method == 'K-Means':
        from sklearn.cluster import KMeans
        # 使用文档-主题分布中的主题数作为聚类数
        n_clusters = doc_topic_dist.shape[1]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(doc_topic_dist)
    elif clustering_method == 'DBSCAN':
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        labels = dbscan.fit_predict(doc_topic_dist)
    elif clustering_method == '层次聚类':
        from sklearn.cluster import AgglomerativeClustering
        # 使用文档-主题分布中的主题数作为聚类数
        n_clusters = doc_topic_dist.shape[1]
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg.fit_predict(doc_topic_dist)
    
    # 绘制散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap=cmap,
        s=marker_size,
        alpha=0.8
    )
    
    # 添加图例
    legend1 = ax.legend(*scatter.legend_elements(),
                        title="主题/聚类")
    ax.add_artist(legend1)
    
    # 计算轮廓系数
    try:
        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(doc_topic_dist, labels)
            ax.set_title(f'文档聚类图 (轮廓系数: {silhouette_avg:.3f})')
        else:
            ax.set_title('文档聚类图')
    except:
        ax.set_title('文档聚类图')
    
    return fig

def render_visualization():
    """渲染可视化模块"""
    st.header("主题模型可视化")
    
    # 初始化会话状态变量
    if "num_topics" not in st.session_state:
        st.session_state.num_topics = 5
    
    if "doc_topic_dist" not in st.session_state:
        st.session_state.doc_topic_dist = None
    
    if "topic_keywords" not in st.session_state:
        st.session_state.topic_keywords = {}
    
    if "lda_model" not in st.session_state:
        st.session_state.lda_model = None
    
    # 检查是否有训练好的模型
    if 'training_complete' not in st.session_state or not st.session_state.training_complete:
        st.warning("请先完成模型训练步骤!")
        return
    
    # ... existing code ...