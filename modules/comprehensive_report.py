# -*- coding: utf-8 -*-
"""
专业学术报告生成器 - 完整版
生成包含所有图表和详细分析的文献报告
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
import base64
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px

from utils.font_config import setup_matplotlib_chinese


def fig_to_base64(fig):
    """将matplotlib图表转换为base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def plotly_to_base64(fig):
    """将Plotly图表转换为静态图片base64"""
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=600)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    except:
        # 如果失败，返回HTML嵌入
        return None


def generate_wordcloud_image(word_freq):
    """生成词云图"""
    try:
        setup_matplotlib_chinese()
        
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttf',
            'fonts/SimHei.ttf',
        ]
        
        wc = None
        for font_path in font_paths:
            try:
                wc = WordCloud(
                    width=1200,
                    height=600,
                    background_color='white',
                    font_path=font_path,
                    max_words=100,
                    relative_scaling=0.5,
                    colormap='viridis',
                    margin=10
                ).generate_from_frequencies(word_freq)
                break
            except:
                continue
        
        if wc is None:
            return None
        
        fig, ax = plt.subplots(figsize=(15, 7.5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('词云图', fontsize=16, pad=20, fontweight='bold')
        plt.tight_layout(pad=0)
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"词云生成错误: {e}")
        return None



def generate_frequency_bar_chart(word_freq, top_n=20):
    """生成词频柱状图"""
    try:
        setup_matplotlib_chinese()
        
        top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:top_n]
        words, freqs = zip(*top_words)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(words)), freqs, color='#3498db', edgecolor='#2c3e50', linewidth=1)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel('频率', fontsize=13, fontweight='bold')
        ax.set_title(f'高频词汇 Top {top_n}', fontsize=15, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for i, (bar, freq) in enumerate(zip(bars, freqs)):
            ax.text(freq + max(freqs)*0.01, i, f'{freq}', va='center', fontsize=10)
        
        plt.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        print(f"柱状图生成错误: {e}")
        return None


def generate_topic_distribution_chart(doc_topic_dist, topic_keywords):
    """生成主题分布饼图"""
    try:
        setup_matplotlib_chinese()
        
        # 计算每个主题的平均概率
        topic_probs = doc_topic_dist.mean(axis=0)
        
        # 准备标签
        labels = []
        for i, prob in enumerate(topic_probs):
            keywords = topic_keywords.get(i, [])[:3]
            label = f"主题{i+1}\n{', '.join(keywords)}" if keywords else f"主题{i+1}"
            labels.append(label)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set3(range(len(topic_probs)))
        wedges, texts, autotexts = ax.pie(
            topic_probs,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 10}
        )
        
        ax.set_title('主题分布', fontsize=15, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"饼图生成错误: {e}")
        return None


def generate_cooccurrence_network_chart(cooccurrence_matrix, top_n=30):
    """生成共现网络图"""
    try:
        import networkx as nx
        setup_matplotlib_chinese()
        
        # 获取top N共现对
        top_pairs = sorted(cooccurrence_matrix.items(), key=lambda x: -x[1])[:top_n]
        
        # 创建网络图
        G = nx.Graph()
        for (word1, word2), weight in top_pairs:
            G.add_edge(word1, word2, weight=weight)
        
        # 计算布局
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 绘制
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 绘制边
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights)
        
        nx.draw_networkx_edges(
            G, pos,
            width=[w/max_weight*5 for w in weights],
            alpha=0.5,
            edge_color='gray',
            ax=ax
        )
        
        # 绘制节点
        node_sizes = [G.degree(node) * 300 for node in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color='#3498db',
            alpha=0.7,
            ax=ax
        )
        
        # 绘制标签
        nx.draw_networkx_labels(
            G, pos,
            font_size=9,
            font_weight='bold',
            ax=ax
        )
        
        ax.set_title(f'词语共现网络 (Top {top_n})', fontsize=15, fontweight='bold', pad=20)
        ax.axis('off')
        plt.tight_layout()
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"网络图生成错误: {e}")
        return None


def generate_temporal_trend_chart(keyword_trends):
    """生成时序趋势图"""
    try:
        setup_matplotlib_chinese()
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for keyword, trend_data in keyword_trends.items():
            if trend_data:
                periods = sorted(trend_data.keys())
                frequencies = [trend_data[p] for p in periods]
                ax.plot(periods, frequencies, marker='o', label=keyword, linewidth=2, markersize=6)
        
        ax.set_xlabel('时间', fontsize=12, fontweight='bold')
        ax.set_ylabel('频率', fontsize=12, fontweight='bold')
        ax.set_title('关键词时序演变趋势', fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"趋势图生成错误: {e}")
        return None


def generate_lda_topic_heatmap(doc_topic_dist, file_names, topic_keywords):
    """生成LDA文档-主题热力图"""
    try:
        setup_matplotlib_chinese()
        
        # 限制显示的文档数量
        max_docs = 30
        if len(doc_topic_dist) > max_docs:
            doc_topic_dist = doc_topic_dist[:max_docs]
            file_names = file_names[:max_docs]
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(file_names) * 0.3)))
        
        im = ax.imshow(doc_topic_dist, cmap='YlOrRd', aspect='auto')
        
        # 设置坐标轴
        ax.set_xticks(range(doc_topic_dist.shape[1]))
        ax.set_yticks(range(len(file_names)))
        
        # 生成主题标签
        topic_labels = []
        for i in range(doc_topic_dist.shape[1]):
            keywords = topic_keywords.get(i, [])[:2]
            label = f"主题{i+1}\n{','.join(keywords)}" if keywords else f"主题{i+1}"
            topic_labels.append(label)
        
        ax.set_xticklabels(topic_labels, fontsize=9)
        ax.set_yticklabels([fn[:20] + '...' if len(fn) > 20 else fn for fn in file_names], fontsize=8)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('主题概率', fontsize=11)
        
        ax.set_title('文档-主题分布热力图', fontsize=15, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"热力图生成错误: {e}")
        return None


def generate_cluster_visualization(cluster_labels, file_names):
    """生成聚类可视化图"""
    try:
        setup_matplotlib_chinese()
        
        # 统计每个聚类的文档数
        from collections import Counter
        cluster_counts = Counter(cluster_labels)
        
        clusters = sorted(cluster_counts.keys())
        counts = [cluster_counts[c] for c in clusters]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(clusters)), counts, color='#2ecc71', edgecolor='#27ae60', linewidth=1.5)
        
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels([f'聚类{c}' for c in clusters], fontsize=11)
        ax.set_ylabel('文档数量', fontsize=12, fontweight='bold')
        ax.set_title('聚类分布', fontsize=15, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        print(f"聚类图生成错误: {e}")
        return None
