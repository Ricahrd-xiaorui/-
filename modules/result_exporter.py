# -*- coding: utf-8 -*-
"""
结果导出模块 - 整合所有分析结果的导出功能
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import base64
from io import BytesIO
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
import pyLDAvis
import pyLDAvis.gensim_models
from utils.session_state import get_session_state, log_message, register_chart, get_all_charts

# 安全导入新模块
def safe_import(module_name, class_name):
    """安全导入模块类"""
    try:
        module = __import__(f'modules.{module_name}', fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        return None

# 导入新模块的类
FrequencyAnalyzer = safe_import('word_frequency_cooccurrence', 'FrequencyAnalyzer')
CooccurrenceAnalyzer = safe_import('word_frequency_cooccurrence', 'CooccurrenceAnalyzer')
TemporalAnalyzer = safe_import('temporal_topic_evolution', 'TemporalAnalyzer')
CitationAnalyzer = safe_import('citation_analyzer', 'CitationAnalyzer')
SemanticNetworkBuilder = safe_import('semantic_network_builder', 'SemanticNetworkBuilder')
QualitativeCoder = safe_import('qualitative_coding', 'QualitativeCoder')
CodingScheme = safe_import('qualitative_coding', 'CodingScheme')


def fig_to_base64(fig):
    """将matplotlib图表转换为base64编码的图片"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def plotly_to_html(fig):
    """将Plotly图表转换为HTML字符串"""
    return fig.to_html(include_plotlyjs='cdn', div_id=None, config={'displayModeBar': False})


def generate_wordcloud_base64(word_freq, font_path=None):
    """生成词云图并转换为base64"""
    try:
        if not word_freq:
            return None
        
        # 尝试多个字体路径
        font_paths = [
            'fonts/SimHei.ttf',
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttf',  # 微软雅黑
            None  # 使用默认字体
        ]
        
        wc = None
        for fp in font_paths:
            try:
                wc = WordCloud(
                    width=1000,
                    height=500,
                    background_color='white',
                    font_path=fp,
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
        
        # 转换为图片
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"词云生成错误: {e}")
        return None


def generate_frequency_chart_base64(word_freq, top_n=20):
    """生成词频柱状图并转换为base64"""
    try:
        if not word_freq:
            return None
        
        # 获取top N词频
        top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:top_n]
        if not top_words:
            return None
            
        words, freqs = zip(*top_words)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建柱状图
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(words)), freqs, color='steelblue', edgecolor='navy', linewidth=0.5)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel('频率', fontsize=12, fontweight='bold')
        ax.set_title(f'高频词汇 Top {top_n}', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 在柱子上添加数值标签
        for i, (bar, freq) in enumerate(zip(bars, freqs)):
            ax.text(freq, i, f' {freq}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"柱状图生成错误: {e}")
        return None
        ax.set_xlabel('频率')
        ax.set_title(f'高频词汇 Top {top_n}')
        ax.grid(axis='x', alpha=0.3)
        
        return fig_to_base64(fig)
    except Exception as e:
        return None


def generate_report_html(file_names, texts, lda_model, topic_keywords, doc_topic_dist, 
                         coherence_score, perplexity, pyldavis_html=None):
    """生成HTML分析报告"""
    coherence_value = f"{coherence_score:.4f}" if coherence_score is not None else "N/A"
    perplexity_value = f"{perplexity:.4f}" if perplexity is not None else "N/A"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LDA主题模型分析报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; max-width: 1200px; margin: 0 auto; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .section {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
            .topic-keywords {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 10px; }}
            .keyword {{ display: inline-block; background-color: #e0f7fa; padding: 3px 8px; margin: 3px; border-radius: 3px; }}
            .metric {{ font-size: 18px; font-weight: bold; color: #0288d1; }}
            .pyldavis-container {{ width: 100%; height: 800px; border: none; }}
        </style>
    </head>
    <body>
        <h1>LDA主题模型分析报告</h1>
        <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="section">
            <h2>1. 分析概述</h2>
            <ul>
                <li>文档数量: {len(texts)}</li>
                <li>主题数量: {lda_model.num_topics}</li>
                <li>连贯性分数: <span class="metric">{coherence_value}</span></li>
                <li>困惑度: <span class="metric">{perplexity_value}</span></li>
            </ul>
        </div>
        
        <div class="section">
            <h2>2. 主题关键词</h2>
    """
    
    for topic_id, keywords in topic_keywords.items():
        html += f'<div class="topic-keywords"><h3>主题 {topic_id + 1}</h3><p>'
        for word in keywords[:20]:
            html += f'<span class="keyword">{word}</span> '
        html += '</p></div>'
    
    html += '</div><div class="section"><h2>3. 文档-主题分布</h2><table><tr><th>文档</th>'
    
    for i in range(lda_model.num_topics):
        html += f"<th>主题 {i+1}</th>"
    html += "</tr>"
    
    if len(doc_topic_dist) > 0:
        for i, file_name in enumerate(file_names[:len(doc_topic_dist)]):
            html += f"<tr><td>{file_name}</td>"
            for j in range(doc_topic_dist.shape[1]):
                html += f"<td>{doc_topic_dist[i, j]:.4f}</td>"
            html += "</tr>"
    
    html += "</table></div>"
    
    if pyldavis_html:
        html += f"""
        <div class="section">
            <h2>4. 交互式主题可视化</h2>
            <iframe class="pyldavis-container" srcdoc='{pyldavis_html.replace("'", "\\'")}'></iframe>
        </div>
        """
    
    html += """
        <footer><p>生成自: 政策文件LDA主题模型可视化分析系统 v2.0</p></footer>
    </body></html>
    """
    
    return html


def generate_comprehensive_report():
    """生成综合分析报告（包含所有已完成的分析）"""
    # 导入报告生成器的图表函数
    try:
        from modules.comprehensive_report import (
            generate_wordcloud_image,
            generate_frequency_bar_chart,
            generate_topic_distribution_chart,
            generate_cooccurrence_network_chart,
            generate_temporal_trend_chart,
            generate_lda_topic_heatmap,
            generate_cluster_visualization
        )
    except ImportError:
        # 如果导入失败，使用本地函数
        generate_wordcloud_image = generate_wordcloud_base64
        generate_frequency_bar_chart = generate_frequency_chart_base64
        generate_topic_distribution_chart = None
        generate_cooccurrence_network_chart = None
        generate_temporal_trend_chart = None
        generate_lda_topic_heatmap = None
        generate_cluster_visualization = None
    
    # 收集所有session_state中的分析结果
    report_sections = []
    
    # 基本信息
    file_names = st.session_state.get("file_names", [])
    texts = st.session_state.get("texts", [])
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>政策文本分析综合报告</title>
        <style>
            body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; line-height: 1.8; margin: 0; padding: 20px; color: #333; max-width: 1400px; margin: 0 auto; background-color: #f5f5f5; }}
            .container {{ background-color: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #1976d2; border-bottom: 3px solid #1976d2; padding-bottom: 15px; margin-bottom: 30px; }}
            h2 {{ color: #0288d1; border-left: 5px solid #0288d1; padding-left: 15px; margin-top: 40px; margin-bottom: 20px; }}
            h3 {{ color: #0097a7; margin-top: 25px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #e3f2fd; color: #1976d2; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .section {{ margin-bottom: 40px; padding: 25px; background-color: #fafafa; border-radius: 8px; }}
            .metric-box {{ display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px 30px; margin: 10px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .metric-value {{ font-size: 32px; font-weight: bold; display: block; }}
            .metric-label {{ font-size: 14px; opacity: 0.9; }}
            .keyword {{ display: inline-block; background-color: #e1f5fe; color: #01579b; padding: 5px 12px; margin: 4px; border-radius: 15px; font-size: 14px; }}
            .warning {{ background-color: #fff3e0; border-left: 4px solid #ff9800; padding: 15px; margin: 15px 0; border-radius: 4px; }}
            .info {{ background-color: #e3f2fd; border-left: 4px solid #2196f3; padding: 15px; margin: 15px 0; border-radius: 4px; }}
            .success {{ background-color: #e8f5e9; border-left: 4px solid #4caf50; padding: 15px; margin: 15px 0; border-radius: 4px; }}
            .toc {{ background-color: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
            .toc ul {{ list-style-type: none; padding-left: 0; }}
            .toc li {{ padding: 8px 0; }}
            .toc a {{ color: #1976d2; text-decoration: none; }}
            .toc a:hover {{ text-decoration: underline; }}
            footer {{ margin-top: 50px; padding-top: 20px; border-top: 2px solid #e0e0e0; text-align: center; color: #666; }}
            .chart-placeholder {{ background-color: #f0f0f0; padding: 40px; text-align: center; color: #666; border-radius: 8px; margin: 20px 0; }}
            .chart-gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
            .chart-item {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .chart-item img {{ max-width: 100%; height: auto; border-radius: 4px; }}
            .chart-item h4 {{ margin: 10px 0 5px 0; color: #1976d2; }}
            .chart-item p {{ margin: 0; color: #666; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 政策文本分析综合报告</h1>
            <p style="color: #666; font-size: 16px;">生成时间: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}</p>
            
            <div class="toc">
                <h3>📑 目录</h3>
                <ul>
                    <li><a href="#overview">1. 分析概览</a></li>
                    <li><a href="#text-stats">2. 文本统计</a></li>
                    <li><a href="#topic-model">3. 主题建模</a></li>
                    <li><a href="#frequency">4. 词频分析</a></li>
                    <li><a href="#cooccurrence">5. 共现分析</a></li>
                    <li><a href="#clustering">6. 聚类分析</a></li>
                    <li><a href="#temporal">7. 时序分析</a></li>
                    <li><a href="#semantic">8. 语义网络</a></li>
                    <li><a href="#charts">9. 图表展示</a></li>
                    <li><a href="#summary">10. 分析总结</a></li>
                </ul>
            </div>
    """
    
    # 1. 分析概览
    html += f"""
            <div class="section" id="overview">
                <h2>1. 分析概览</h2>
                <div style="text-align: center;">
                    <div class="metric-box">
                        <span class="metric-value">{len(file_names)}</span>
                        <span class="metric-label">文档数量</span>
                    </div>
                    <div class="metric-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                        <span class="metric-value">{len(texts)}</span>
                        <span class="metric-label">已处理文档</span>
                    </div>
                    <div class="metric-box" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                        <span class="metric-value">{sum(len(text) for text in texts) if texts else 0}</span>
                        <span class="metric-label">总词数</span>
                    </div>
                </div>
            </div>
    """
    
    # 2. 文本统计
    text_stats = st.session_state.get("text_statistics")
    if text_stats is not None:
        try:
            if hasattr(text_stats, 'empty') and not text_stats.empty:
                html += f"""
            <div class="section" id="text-stats">
                <h2>2. 文本统计</h2>
                <p>对所有文档的基本统计信息：</p>
                {text_stats.to_html(index=False, classes='data-table')}
            </div>
                """
        except:
            pass
    
    # 3. 主题建模
    lda_model = st.session_state.get("lda_model")
    topic_keywords = st.session_state.get("topic_keywords", {})
    doc_topic_dist = st.session_state.get("doc_topic_dist")
    
    if lda_model and topic_keywords:
        coherence_score = st.session_state.get('coherence_score')
        perplexity = st.session_state.get('perplexity')
        
        html += f"""
            <div class="section" id="topic-model">
                <h2>3. 主题建模分析</h2>
                <div class="info">
                    <strong>模型参数：</strong>
                    <ul>
                        <li>主题数量: {lda_model.num_topics}</li>
                        <li>连贯性分数: {f'{coherence_score:.4f}' if coherence_score else 'N/A'}</li>
                        <li>困惑度: {f'{perplexity:.4f}' if perplexity else 'N/A'}</li>
                    </ul>
                </div>
                
                <h3>3.1 主题关键词</h3>
                <p>以下是通过LDA模型提取的各主题关键词，反映了文本集合的主要主题分布：</p>
        """
        
        for topic_id, keywords in topic_keywords.items():
            html += f"""
                <div style="margin: 15px 0; padding: 15px; background-color: #f8f9fa; border-left: 4px solid #0288d1; border-radius: 4px;">
                    <strong style="color: #0288d1; font-size: 16px;">主题 {topic_id + 1}:</strong><br>
                    <div style="margin-top: 10px;">
            """
            for word in keywords[:15]:
                html += f'<span class="keyword">{word}</span> '
            html += "</div></div>"
        
        # 添加主题分布饼图
        if doc_topic_dist is not None and generate_topic_distribution_chart:
            try:
                topic_pie_img = generate_topic_distribution_chart(doc_topic_dist, topic_keywords)
                if topic_pie_img:
                    html += f"""
                <h3>3.2 主题分布图</h3>
                <p>各主题在所有文档中的平均分布情况：</p>
                <div style="text-align: center; margin: 20px 0;">
                    <img src="{topic_pie_img}" style="max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                </div>
                    """
            except Exception as e:
                print(f"主题分布图生成失败: {e}")
        
        # 添加文档-主题热力图
        if doc_topic_dist is not None and generate_lda_topic_heatmap:
            try:
                heatmap_img = generate_lda_topic_heatmap(doc_topic_dist, file_names, topic_keywords)
                if heatmap_img:
                    html += f"""
                <h3>3.3 文档-主题分布热力图</h3>
                <p>展示每个文档在各主题上的概率分布（显示前30个文档）：</p>
                <div style="text-align: center; margin: 20px 0;">
                    <img src="{heatmap_img}" style="max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                </div>
                    """
            except Exception as e:
                print(f"热力图生成失败: {e}")
        
        # 添加PyLDAvis可视化
        pyldavis_html = st.session_state.get("pyldavis_html")
        if pyldavis_html:
            html += f"""
                <h3>3.4 交互式主题可视化 (PyLDAvis)</h3>
                <p>交互式探索主题间的关系和主题内的词语分布：</p>
                <div style="margin: 20px 0;">
                    <iframe style="width: 100%; height: 800px; border: 1px solid #ddd; border-radius: 8px;" 
                            srcdoc='{pyldavis_html.replace("'", "\\'")}'></iframe>
                </div>
            """
        
        html += "</div>"
    
    # 4. 词频分析
    word_freq = st.session_state.get("word_frequencies")
    if word_freq:
        top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:20]
        
        html += f"""
            <div class="section" id="frequency">
                <h2>4. 词频分析</h2>
                <p>通过词频统计识别文本中的高频词汇，揭示核心概念和关注重点。</p>
        """
        
        # 生成词云图
        try:
            wordcloud_img = generate_wordcloud_image(word_freq)
            if wordcloud_img:
                html += f"""
                <h3>4.1 词云图</h3>
                <p>词云直观展示高频词汇，词语大小反映其出现频率：</p>
                <div style="text-align: center; margin: 20px 0;">
                    <img src="{wordcloud_img}" style="max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                </div>
                """
        except Exception as e:
            print(f"词云生成失败: {e}")
        
        # 生成词频柱状图
        try:
            freq_chart_img = generate_frequency_bar_chart(word_freq, top_n=20)
            if freq_chart_img:
                html += f"""
                <h3>4.2 高频词汇柱状图</h3>
                <p>Top 20高频词汇及其出现次数：</p>
                <div style="text-align: center; margin: 20px 0;">
                    <img src="{freq_chart_img}" style="max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                </div>
                """
        except Exception as e:
            print(f"柱状图生成失败: {e}")
        
        html += """
                <h3>4.3 高频词汇表 (Top 20)</h3>
                <table>
                    <tr><th>排名</th><th>词语</th><th>频率</th><th>占比</th></tr>
        """
        total_freq = sum(word_freq.values())
        for i, (word, freq) in enumerate(top_words, 1):
            percentage = (freq / total_freq * 100) if total_freq > 0 else 0
            html += f"<tr><td>{i}</td><td><strong>{word}</strong></td><td>{freq}</td><td>{percentage:.2f}%</td></tr>"
        html += "</table></div>"
    
    # 5. 共现分析
    cooccurrence_matrix = st.session_state.get("cooccurrence_matrix")
    if cooccurrence_matrix:
        top_pairs = sorted(cooccurrence_matrix.items(), key=lambda x: -x[1])[:15]
        html += f"""
            <div class="section" id="cooccurrence">
                <h2>5. 词语共现分析</h2>
                <p>共现分析揭示词语之间的关联关系，共找到 <strong>{len(cooccurrence_matrix)}</strong> 对共现词语。</p>
        """
        
        # 添加共现网络图
        if generate_cooccurrence_network_chart:
            try:
                network_img = generate_cooccurrence_network_chart(cooccurrence_matrix, top_n=30)
                if network_img:
                    html += f"""
                <h3>5.1 共现网络图</h3>
                <p>展示高频共现词对的网络关系（Top 30），节点大小表示连接数，边的粗细表示共现强度：</p>
                <div style="text-align: center; margin: 20px 0;">
                    <img src="{network_img}" style="max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                </div>
                    """
            except Exception as e:
                print(f"网络图生成失败: {e}")
        
        html += """
                <h3>5.2 高频共现词对 (Top 15)</h3>
                <table>
                    <tr><th>排名</th><th>词语1</th><th>词语2</th><th>共现次数</th></tr>
        """
        for i, ((word1, word2), count) in enumerate(top_pairs, 1):
            html += f"<tr><td>{i}</td><td><strong>{word1}</strong></td><td><strong>{word2}</strong></td><td>{count}</td></tr>"
        html += "</table></div>"
    
    # 6. 聚类分析
    cluster_labels = st.session_state.get("cluster_labels")
    if cluster_labels is not None:
        n_clusters = len(set(cluster_labels))
        html += f"""
            <div class="section" id="clustering">
                <h2>6. 聚类分析</h2>
                <p>通过聚类算法将文档分为 <strong>{n_clusters}</strong> 个聚类，发现文档间的相似性模式。</p>
                <div class="info">
                    <strong>聚类方法:</strong> {st.session_state.get('clustering_method', 'K-Means')}
                </div>
        """
        
        # 添加聚类分布图
        if generate_cluster_visualization:
            try:
                cluster_img = generate_cluster_visualization(cluster_labels, file_names)
                if cluster_img:
                    html += f"""
                <h3>6.1 聚类分布图</h3>
                <p>各聚类包含的文档数量分布：</p>
                <div style="text-align: center; margin: 20px 0;">
                    <img src="{cluster_img}" style="max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                </div>
                    """
            except Exception as e:
                print(f"聚类图生成失败: {e}")
        
        html += "</div>"
    
    # 7. 时序分析
    time_labels = st.session_state.get("time_labels")
    keyword_trends = st.session_state.get("keyword_trends")
    if time_labels:
        html += f"""
            <div class="section" id="temporal">
                <h2>7. 时序演变分析</h2>
                <p>分析文本内容随时间的演变趋势，已标注时间标签的文档数: <strong>{len(time_labels)}</strong></p>
                <div class="info">
                    <strong>时间范围:</strong> {min(time_labels.values())} - {max(time_labels.values())}
                </div>
        """
        
        # 添加时序趋势图
        if keyword_trends and generate_temporal_trend_chart:
            try:
                trend_img = generate_temporal_trend_chart(keyword_trends)
                if trend_img:
                    html += f"""
                <h3>7.1 关键词时序演变趋势</h3>
                <p>关键词在不同时间段的频率变化：</p>
                <div style="text-align: center; margin: 20px 0;">
                    <img src="{trend_img}" style="max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                </div>
                    """
            except Exception as e:
                print(f"趋势图生成失败: {e}")
        
        html += "</div>"
    
    # 8. 语义网络
    semantic_network = st.session_state.get("semantic_network")
    if semantic_network:
        num_nodes = semantic_network.number_of_nodes()
        num_edges = semantic_network.number_of_edges()
        
        # 计算网络统计指标
        try:
            import networkx as nx
            density = nx.density(semantic_network)
            avg_degree = sum(dict(semantic_network.degree()).values()) / num_nodes if num_nodes > 0 else 0
        except:
            density = 0
            avg_degree = 0
        
        html += f"""
            <div class="section" id="semantic">
                <h2>8. 语义网络分析</h2>
                <p>构建词语间的语义关联网络，揭示概念之间的复杂关系。</p>
                <div style="text-align: center;">
                    <div class="metric-box">
                        <span class="metric-value">{num_nodes}</span>
                        <span class="metric-label">网络节点数</span>
                    </div>
                    <div class="metric-box" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                        <span class="metric-value">{num_edges}</span>
                        <span class="metric-label">网络边数</span>
                    </div>
                    <div class="metric-box" style="background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);">
                        <span class="metric-value">{avg_degree:.1f}</span>
                        <span class="metric-label">平均度数</span>
                    </div>
                </div>
                <div class="info" style="margin-top: 20px;">
                    <strong>网络密度:</strong> {density:.4f} (值越大表示网络连接越紧密)
                </div>
            </div>
        """
    
    # 9. 图表展示
    charts = get_all_charts()
    if charts:
        html += f"""
            <div class="section" id="charts">
                <h2>9. 图表展示</h2>
                <p>以下是所有已生成的可视化图表：</p>
                <div class="chart-gallery">
        """
        for chart_id, chart_info in charts.items():
            img_data = chart_info.get('image_base64', '')
            if img_data:
                html += f"""
                    <div class="chart-item">
                        <h4>{chart_info.get('title', chart_id)}</h4>
                        <img src="{img_data}" alt="{chart_info.get('title', '')}">
                        <p>{chart_info.get('description', '')}</p>
                        <p style="font-size: 12px; color: #999;">类型: {chart_info.get('chart_type', 'unknown')} | 时间: {chart_info.get('timestamp', '')}</p>
                    </div>
                """
        html += """
                </div>
            </div>
        """
    
    # 10. 分析总结
    html += f"""
            <div class="section" id="summary">
                <h2>10. 分析总结与建议</h2>
                
                <h3>9.1 已完成的分析</h3>
                <div class="success">
                    <ul>
    """
    
    completed_analyses = []
    if texts: completed_analyses.append("文本预处理与分词")
    if text_stats is not None: completed_analyses.append("文本统计分析")
    if lda_model: completed_analyses.append("LDA主题建模")
    if word_freq: completed_analyses.append("词频统计分析")
    if cooccurrence_matrix: completed_analyses.append("词语共现分析")
    if cluster_labels is not None: completed_analyses.append("文档聚类分析")
    if time_labels: completed_analyses.append("时序演变分析")
    if semantic_network: completed_analyses.append("语义网络构建")
    
    for analysis in completed_analyses:
        html += f"<li><strong>{analysis}</strong></li>"
    
    html += f"""
                    </ul>
                    <p style="font-size: 18px; margin-top: 15px;"><strong>✅ 总计完成 {len(completed_analyses)} 项分析</strong></p>
                </div>
                
                <h3>9.2 主要发现</h3>
                <div class="info">
                    <ul>
    """
    
    # 根据分析结果生成主要发现
    if lda_model and topic_keywords:
        html += f"<li><strong>主题分布：</strong>识别出 {lda_model.num_topics} 个主要主题，涵盖文本集合的核心内容</li>"
    
    if word_freq:
        top_word = max(word_freq.items(), key=lambda x: x[1])
        html += f"<li><strong>高频词汇：</strong>最高频词为「{top_word[0]}」，出现 {top_word[1]} 次</li>"
    
    if cooccurrence_matrix:
        html += f"<li><strong>词语关联：</strong>发现 {len(cooccurrence_matrix)} 对共现词语，揭示概念间的关联模式</li>"
    
    if cluster_labels is not None:
        html += f"<li><strong>文档分组：</strong>文档被分为 {len(set(cluster_labels))} 个聚类，体现内容的多样性</li>"
    
    if semantic_network:
        html += f"<li><strong>语义网络：</strong>构建了包含 {semantic_network.number_of_nodes()} 个节点的语义网络</li>"
    
    html += """
                    </ul>
                </div>
                
                <h3>9.3 研究建议</h3>
                <div class="warning">
                    <ul>
                        <li>建议结合领域知识对主题进行深入解读和命名</li>
                        <li>可以调整主题数量参数，探索不同粒度的主题划分</li>
                        <li>关注高频词和共现词对，它们往往揭示核心概念</li>
                        <li>利用时序分析追踪概念演变，发现趋势变化</li>
                        <li>通过语义网络识别关键节点和社区结构</li>
                    </ul>
                </div>
            </div>
            
            <footer>
                <p>本报告由 <strong>政策文件LDA主题模型可视化分析系统 v2.0</strong> 自动生成</p>
                <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p style="margin-top: 10px; color: #666;">
                    <em>注：本报告基于自动化文本分析生成，建议结合人工解读和领域知识进行综合判断</em>
                </p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    return html


def dataframe_to_csv(df):
    """将DataFrame转换为CSV字符串"""
    return df.to_csv(index=False).encode('utf-8-sig')


def render_comprehensive_report():
    """渲染综合分析报告"""
    st.subheader("📋 综合分析报告")
    
    st.markdown("""
    生成包含所有已完成分析的完整报告，包括：
    - 文本统计
    - 主题建模
    - 词频分析
    - 共现分析
    - 聚类分析
    - 时序分析
    - 语义网络
    - 其他高级分析
    """)
    
    # 检查是否有分析结果
    has_results = False
    results_summary = []
    
    if st.session_state.get("texts"):
        has_results = True
        results_summary.append("✅ 文本预处理")
    
    if st.session_state.get("text_statistics") is not None:
        results_summary.append("✅ 文本统计")
    
    if st.session_state.get("lda_model"):
        results_summary.append("✅ 主题建模")
    
    if st.session_state.get("word_frequencies"):
        results_summary.append("✅ 词频分析")
    
    if st.session_state.get("cooccurrence_matrix"):
        results_summary.append("✅ 共现分析")
    
    if st.session_state.get("cluster_labels") is not None:
        results_summary.append("✅ 聚类分析")
    
    if st.session_state.get("time_labels"):
        results_summary.append("✅ 时序分析")
    
    if st.session_state.get("semantic_network"):
        results_summary.append("✅ 语义网络")
    
    if not has_results:
        st.warning("⚠️ 尚未进行任何分析，请先完成文本预处理和相关分析")
        return
    
    # 显示已完成的分析
    st.info(f"📊 已完成 {len(results_summary)} 项分析")
    
    col1, col2 = st.columns(2)
    with col1:
        for item in results_summary[:len(results_summary)//2 + 1]:
            st.write(item)
    with col2:
        for item in results_summary[len(results_summary)//2 + 1:]:
            st.write(item)
    
    st.markdown("---")
    
    # 报告选项
    st.markdown("### 报告选项")
    
    report_col1, report_col2, report_col3 = st.columns(3)
    
    with report_col1:
        include_details = st.checkbox("包含详细数据表格", value=True, help="包含完整的数据表格")
    
    with report_col2:
        include_charts = st.checkbox("包含可视化图表", value=True, help="包含词云、柱状图等图表")
    
    with report_col3:
        detailed_mode = st.checkbox("学术报告模式", value=True, help="生成详细的学术风格报告，包含分析说明")
    
    # 生成报告按钮
    if st.button("📄 生成综合报告", type="primary", use_container_width=True):
        with st.spinner("正在生成综合报告，请稍候..."):
            try:
                # 生成报告
                html_content = generate_comprehensive_report()
                
                # 提供下载
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"综合分析报告_{timestamp}.html"
                
                st.download_button(
                    label="📥 下载综合报告 (HTML)",
                    data=html_content.encode('utf-8'),
                    file_name=filename,
                    mime="text/html",
                    use_container_width=True
                )
                
                st.success(f"✅ 综合报告生成成功！包含 {len(results_summary)} 项分析结果")
                
                # 显示预览
                with st.expander("📖 报告预览", expanded=False):
                    st.components.v1.html(html_content, height=600, scrolling=True)
                
                log_message(f"生成综合分析报告，包含{len(results_summary)}项分析")
                
            except Exception as e:
                st.error(f"❌ 生成报告时出错：{str(e)}")
                import traceback
                with st.expander("查看详细错误信息"):
                    st.code(traceback.format_exc())
    
    # 提示信息
    st.markdown("---")
    st.info("""
    💡 **使用提示**：
    - **学术报告模式**：生成详细的分析说明和解读，适合论文、报告使用
    - **包含图表**：自动生成词云和柱状图（如果生成失败会跳过）
    - 报告包含交互式目录，方便导航
    - 建议完成更多分析后再生成报告，以获得更全面的结果
    - 生成时间约5-15秒，请耐心等待
    """)


def dataframe_to_csv(df):
    """将DataFrame转换为CSV字符串"""
    return df.to_csv(index=False).encode('utf-8-sig')


def render_exporter():
    """渲染结果导出模块"""
    st.header("💾 结果导出")
    
    with st.expander("📖 功能介绍", expanded=False):
        st.markdown("""
        ### 导出内容
        
        | 类别 | 导出内容 | 文件格式 |
        |------|----------|----------|
        | 综合报告 | 所有分析结果的完整报告 | HTML/PDF |
        | 主题分析 | 分析报告、主题关键词、文档-主题分布 | HTML/CSV |
        | 基础分析 | 文本统计、词频表、共现矩阵 | CSV |
        | 高级分析 | 聚类、时序、比较、引用、语义网络、编码 | CSV |
        | 模型文件 | LDA模型 | ZIP |
        
        **使用说明**：CSV文件使用UTF-8-BOM编码，兼容Excel
        """)
    
    # 创建选项卡
    export_tabs = st.tabs([
        "📋 综合报告",
        "📊 主题分析", 
        "📈 基础分析", 
        "🔬 高级分析",
        "💾 模型导出"
    ])
    
    with export_tabs[0]:
        render_comprehensive_report()
    
    with export_tabs[1]:
        render_topic_export()
    
    with export_tabs[2]:
        render_basic_analysis_export()
    
    with export_tabs[3]:
        render_advanced_analysis_export()
    
    with export_tabs[4]:
        render_model_export()


def render_topic_export():
    """渲染主题分析导出"""
    st.subheader("📊 主题分析结果导出")
    
    if not st.session_state.get("training_complete") or not st.session_state.get("lda_model"):
        st.warning('请先在"主题建模"标签页中完成LDA模型训练')
        return
    
    topic_tabs = st.tabs(["分析报告", "数据表格"])
    
    with topic_tabs[0]:
        st.markdown("#### 导出完整分析报告")
        
        report_format = st.radio("报告格式", ["HTML", "PDF"], horizontal=True, key="report_format_radio")
        
        st.write("包含内容:")
        include_topics = st.checkbox("主题关键词", value=True, key="include_topics_checkbox")
        include_doc_dist = st.checkbox("文档-主题分布", value=True, key="include_doc_dist_checkbox")
        include_pyldavis = st.checkbox("PyLDAvis可视化", value=True, key="include_pyldavis_checkbox")
        
        if st.button("生成分析报告", key="generate_report"):
            with st.spinner("正在生成报告..."):
                try:
                    pyldavis_html = st.session_state.get("pyldavis_html") if include_pyldavis else None
                    
                    html_report = generate_report_html(
                        st.session_state.file_names,
                        st.session_state.texts,
                        st.session_state.lda_model,
                        st.session_state.topic_keywords if include_topics else {},
                        st.session_state.doc_topic_dist if include_doc_dist else np.array([]),
                        st.session_state.get("coherence_score"),
                        st.session_state.get("perplexity"),
                        pyldavis_html
                    )
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    st.download_button(
                        label="📥 下载HTML报告",
                        data=html_report.encode('utf-8'),
                        file_name=f"lda_report_{timestamp}.html",
                        mime="text/html"
                    )
                    
                    if report_format == "PDF":
                        st.info("提示: 可用浏览器打印功能将HTML转换为PDF")
                
                except Exception as e:
                    st.error(f"生成报告时出错: {str(e)}")
    
    with topic_tabs[1]:
        st.markdown("#### 导出数据表格")
        
        data_type = st.selectbox("选择数据", ["主题关键词", "文档-主题分布", "主题相似度矩阵"], key="export_topic_data_type")
        
        if data_type == "主题关键词" and st.session_state.get("topic_keywords"):
            all_keywords = {}
            max_keywords = 0
            for topic_id, keywords in st.session_state.topic_keywords.items():
                all_keywords[f"主题{topic_id+1}"] = keywords
                max_keywords = max(max_keywords, len(keywords))
            
            for topic, keywords in all_keywords.items():
                if len(keywords) < max_keywords:
                    all_keywords[topic] = keywords + [""] * (max_keywords - len(keywords))
            
            df = pd.DataFrame(all_keywords)
            st.dataframe(df.head(10), width='stretch')
            
            st.download_button("📥 下载CSV", dataframe_to_csv(df), 
                             f"topic_keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        
        elif data_type == "文档-主题分布" and st.session_state.get("doc_topic_dist") is not None:
            topics = [f"主题{i+1}" for i in range(st.session_state.doc_topic_dist.shape[1])]
            df = pd.DataFrame(st.session_state.doc_topic_dist, columns=topics)
            df.insert(0, '文档', st.session_state.file_names[:len(df)])
            st.dataframe(df.head(10), width='stretch')
            
            st.download_button("📥 下载CSV", dataframe_to_csv(df),
                             f"doc_topic_dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        
        elif data_type == "主题相似度矩阵":
            try:
                num_topics = st.session_state.lda_model.num_topics
                topic_vectors = []
                for i in range(num_topics):
                    topic_vector = [0] * len(st.session_state.dictionary)
                    for word_id, weight in st.session_state.lda_model.get_topic_terms(i, topn=len(st.session_state.dictionary)):
                        topic_vector[word_id] = weight
                    topic_vectors.append(topic_vector)
                
                topic_vectors = np.array(topic_vectors)
                similarity_matrix = np.zeros((num_topics, num_topics))
                
                for i in range(num_topics):
                    for j in range(num_topics):
                        if i == j:
                            similarity_matrix[i, j] = 1.0
                        else:
                            dot_product = np.dot(topic_vectors[i], topic_vectors[j])
                            norm_i = np.linalg.norm(topic_vectors[i])
                            norm_j = np.linalg.norm(topic_vectors[j])
                            if norm_i > 0 and norm_j > 0:
                                similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
                
                topics = [f"主题{i+1}" for i in range(num_topics)]
                df = pd.DataFrame(similarity_matrix, index=topics, columns=topics)
                st.dataframe(df, width='stretch')
                
                st.download_button("📥 下载CSV", dataframe_to_csv(df.reset_index().rename(columns={"index": "主题"})),
                                 f"topic_similarity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
            except Exception as e:
                st.error(f"计算相似度矩阵时出错: {str(e)}")
        else:
            st.warning("没有可用数据")


def render_basic_analysis_export():
    """渲染基础分析导出"""
    st.subheader("📈 基础分析结果导出")
    
    if not st.session_state.get("texts"):
        st.warning('请先在"文本预处理"标签页中完成文本预处理')
        return
    
    data_type = st.selectbox("选择数据", ["文本统计", "词频表", "共现矩阵"], key="export_basic_data_type")
    
    if data_type == "文本统计":
        try:
            from modules.text_analysis_readability import create_multi_doc_statistics
            
            raw_texts = st.session_state.get("raw_texts", [])
            texts = st.session_state.get("texts", [])
            file_names = st.session_state.get("file_names", [])
            
            if raw_texts and texts:
                all_stats = create_multi_doc_statistics(raw_texts, texts, file_names)
                csv_content = all_stats.export_comparison()
                
                df = pd.read_csv(BytesIO(csv_content.encode('utf-8-sig')))
                st.dataframe(df, width='stretch')
                
                st.download_button("📥 下载文本统计CSV", csv_content,
                                 f"text_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
            else:
                st.warning("没有可用的文本数据")
        except Exception as e:
            st.error(f"导出文本统计时出错: {str(e)}")
    
    elif data_type == "词频表":
        try:
            texts = st.session_state.get("texts", [])
            pos_tags = st.session_state.get("pos_tags", [])
            
            if texts and FrequencyAnalyzer:
                analyzer = FrequencyAnalyzer(texts, pos_tags if pos_tags else None)
                csv_content = analyzer.export_frequency_csv(include_pos=bool(pos_tags))
                
                df = pd.read_csv(BytesIO(csv_content.encode('utf-8-sig')))
                st.dataframe(df.head(20), width='stretch')
                
                st.download_button("📥 下载词频表CSV", csv_content,
                                 f"word_frequency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
            else:
                st.warning("没有可用的分词数据")
        except Exception as e:
            st.error(f"导出词频表时出错: {str(e)}")
    
    elif data_type == "共现矩阵":
        try:
            texts = st.session_state.get("texts", [])
            
            if texts and CooccurrenceAnalyzer:
                window_size = st.slider("共现窗口大小", 2, 10, 5, key="export_cooc_window")
                min_freq = st.slider("最小共现频率", 1, 10, 2, key="export_cooc_min_freq")
                
                analyzer = CooccurrenceAnalyzer(texts, window_size)
                analyzer.calculate_cooccurrence()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_content = analyzer.export_matrix_csv(min_freq)
                    st.download_button("📥 下载共现列表CSV", csv_content,
                                     f"cooccurrence_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
                
                with col2:
                    adj_csv = analyzer.export_adjacency_matrix_csv(min_freq, 50)
                    if adj_csv:
                        st.download_button("📥 下载邻接矩阵CSV", adj_csv,
                                         f"cooccurrence_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
            else:
                st.warning("没有可用的分词数据")
        except Exception as e:
            st.error(f"导出共现矩阵时出错: {str(e)}")


def render_advanced_analysis_export():
    """渲染高级分析导出"""
    st.subheader("🔬 高级分析结果导出")
    
    if not st.session_state.get("texts"):
        st.warning('请先在"文本预处理"标签页中完成文本预处理')
        return
    
    data_type = st.selectbox("选择数据", 
                            ["聚类结果", "时序分析", "比较分析", "引用分析", "语义网络", "质性编码"],
                            key="export_advanced_data_type")
    
    if data_type == "聚类结果":
        render_clustering_export()
    elif data_type == "时序分析":
        render_temporal_export()
    elif data_type == "比较分析":
        render_comparative_export()
    elif data_type == "引用分析":
        render_citation_export()
    elif data_type == "语义网络":
        render_semantic_export()
    elif data_type == "质性编码":
        render_coding_export()


def render_clustering_export():
    """渲染聚类结果导出"""
    cluster_labels = st.session_state.get("cluster_labels")
    classification_labels = st.session_state.get("classification_labels", {})
    file_names = st.session_state.get("file_names", [])
    
    if cluster_labels is not None:
        st.markdown("**聚类结果**")
        data = {"文档": file_names[:len(cluster_labels)], "聚类ID": cluster_labels.tolist()}
        df = pd.DataFrame(data)
        st.dataframe(df, width='stretch')
        
        st.download_button("📥 下载聚类结果CSV", dataframe_to_csv(df),
                         f"clustering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    elif classification_labels:
        st.markdown("**分类结果**")
        data = {"文档": list(classification_labels.keys()), "分类标签": list(classification_labels.values())}
        df = pd.DataFrame(data)
        st.dataframe(df, width='stretch')
        
        st.download_button("📥 下载分类结果CSV", dataframe_to_csv(df),
                         f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    else:
        st.info("请先在「高级研究分析 → 聚类分类」中完成聚类或分类分析")


def render_temporal_export():
    """渲染时序分析导出"""
    time_labels = st.session_state.get("time_labels", {})
    
    if time_labels and TemporalAnalyzer:
        texts = st.session_state.get("texts", [])
        file_names = st.session_state.get("file_names", [])
        
        analyzer = TemporalAnalyzer(texts, file_names)
        for doc, label in time_labels.items():
            analyzer.set_time_label(doc, label)
        
        st.markdown("**时间标签数据**")
        csv_content = analyzer.export_time_labels()
        
        df = pd.read_csv(BytesIO(csv_content.encode('utf-8-sig')))
        st.dataframe(df, width='stretch')
        
        st.download_button("📥 下载时间标签CSV", csv_content,
                         f"time_labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        
        keyword_trends = st.session_state.get("keyword_trends")
        if keyword_trends:
            keywords = list(keyword_trends.keys())
            trend_csv = analyzer.export_trend_data(keywords)
            st.download_button("📥 下载关键词趋势CSV", trend_csv,
                             f"keyword_trends_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    else:
        st.info("请先在「高级研究分析 → 时序分析」中设置时间标签")


def render_comparative_export():
    """渲染比较分析导出"""
    sim_matrix_csv = st.session_state.get("sim_matrix_csv")
    comparison_csv = st.session_state.get("comparison_csv")
    
    if sim_matrix_csv or comparison_csv:
        col1, col2 = st.columns(2)
        
        with col1:
            if sim_matrix_csv:
                st.download_button("📥 下载相似度矩阵CSV", sim_matrix_csv,
                                 f"similarity_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        
        with col2:
            if comparison_csv:
                st.download_button("📥 下载比较报告CSV", comparison_csv,
                                 f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    else:
        st.info("请先在「高级研究分析 → 比较分析」中完成比较分析")


def render_citation_export():
    """渲染引用分析导出"""
    citation_network = st.session_state.get("citation_network")
    
    if citation_network and CitationAnalyzer:
        raw_texts = st.session_state.get("raw_texts", [])
        file_names = st.session_state.get("file_names", [])
        
        analyzer = CitationAnalyzer(raw_texts, file_names)
        analyzer.extract_citations()
        analyzer.build_citation_network()
        
        col1, col2 = st.columns(2)
        
        with col1:
            citation_csv = analyzer.export_citation_list()
            st.download_button("📥 下载引用列表CSV", citation_csv,
                             f"citation_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        
        with col2:
            network_csv = analyzer.export_network_data()
            st.download_button("📥 下载引用网络CSV", network_csv,
                             f"citation_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        
        stats_csv = analyzer.export_citation_stats()
        st.download_button("📥 下载引用统计CSV", stats_csv,
                         f"citation_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    else:
        st.info("请先在「高级研究分析 → 引用分析」中完成引用分析")


def render_semantic_export():
    """渲染语义网络导出"""
    semantic_network = st.session_state.get("semantic_network")
    
    if semantic_network and SemanticNetworkBuilder:
        texts = st.session_state.get("texts", [])
        cooccurrence_data = st.session_state.get("cooccurrence_matrix", {})
        
        if cooccurrence_data:
            builder = SemanticNetworkBuilder(texts, cooccurrence_data)
            builder.network = semantic_network
            
            nodes_csv, edges_csv = builder.export_network()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button("📥 下载节点列表CSV", nodes_csv,
                                 f"semantic_nodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
            
            with col2:
                st.download_button("📥 下载边列表CSV", edges_csv,
                                 f"semantic_edges_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        else:
            st.info("请先在「基础文本分析 → 词语共现」中计算共现关系")
    else:
        st.info("请先在「高级研究分析 → 语义网络」中构建语义网络")


def render_coding_export():
    """渲染质性编码导出"""
    coding_scheme = st.session_state.get("coding_scheme")
    coded_segments = st.session_state.get("coded_segments", [])
    
    if coding_scheme and QualitativeCoder and CodingScheme:
        # 如果coding_scheme是字典，需要转换为CodingScheme对象
        if isinstance(coding_scheme, dict):
            scheme = CodingScheme()
            scheme.from_dict(coding_scheme)
        else:
            scheme = coding_scheme
        
        coder = QualitativeCoder(scheme)
        coder.segments = coded_segments
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = coder.export_to_csv()
            st.download_button("📥 导出编码结果CSV", csv_data,
                             f"coding_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        
        with col2:
            stats_csv = coder.export_statistics_csv()
            st.download_button("📥 导出统计数据CSV", stats_csv,
                             f"coding_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    else:
        st.info("请先在「高级研究分析 → 质性编码」中创建编码方案")


def render_model_export():
    """渲染模型导出"""
    st.subheader("💾 导出LDA模型")
    
    if st.session_state.get("model_path"):
        st.write(f"当前模型路径: {st.session_state.model_path}")
        
        if st.button("导出模型文件", key="export_model"):
            with st.spinner("正在准备模型文件..."):
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        model_files = [
                            f"{st.session_state.model_path}.gensim",
                            f"{st.session_state.model_path}.pkl"
                        ]
                        
                        zip_path = os.path.join(temp_dir, "lda_model.zip")
                        
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for file in model_files:
                                if os.path.exists(file):
                                    zipf.write(file, os.path.basename(file))
                        
                        with open(zip_path, "rb") as f:
                            zip_data = f.read()
                        
                        st.download_button("📥 下载模型文件", zip_data,
                                         f"lda_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                         "application/zip")
                        
                        st.success("模型文件已准备好")
                
                except Exception as e:
                    st.error(f"导出模型时出错: {str(e)}")
    else:
        st.warning("未找到保存的模型文件，请在模型训练完成后再尝试导出")
