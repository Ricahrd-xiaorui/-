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
from utils.session_state import get_session_state, log_message

# 安全导入新模块
def safe_import(module_name, class_name):
    """安全导入模块类"""
    try:
        module = __import__(f'modules.{module_name}', fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        return None

# 导入新模块的类
FrequencyAnalyzer = safe_import('frequency_analyzer', 'FrequencyAnalyzer')
CooccurrenceAnalyzer = safe_import('frequency_analyzer', 'CooccurrenceAnalyzer')
TemporalAnalyzer = safe_import('temporal_analyzer', 'TemporalAnalyzer')
CitationAnalyzer = safe_import('citation_analyzer', 'CitationAnalyzer')
SemanticNetworkBuilder = safe_import('semantic_network', 'SemanticNetworkBuilder')
QualitativeCoder = safe_import('qualitative_coding', 'QualitativeCoder')
CodingScheme = safe_import('qualitative_coding', 'CodingScheme')


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
        | 主题分析 | 分析报告、主题关键词、文档-主题分布 | HTML/CSV |
        | 基础分析 | 文本统计、词频表、共现矩阵 | CSV |
        | 高级分析 | 聚类、时序、比较、引用、语义网络、编码 | CSV |
        | 模型文件 | LDA模型 | ZIP |
        
        **使用说明**：CSV文件使用UTF-8-BOM编码，兼容Excel
        """)
    
    # 创建选项卡
    export_tabs = st.tabs([
        "📊 主题分析", 
        "📈 基础分析", 
        "🔬 高级分析",
        "💾 模型导出"
    ])
    
    with export_tabs[0]:
        render_topic_export()
    
    with export_tabs[1]:
        render_basic_analysis_export()
    
    with export_tabs[2]:
        render_advanced_analysis_export()
    
    with export_tabs[3]:
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
            st.dataframe(df.head(10), use_container_width=True)
            
            st.download_button("📥 下载CSV", dataframe_to_csv(df), 
                             f"topic_keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        
        elif data_type == "文档-主题分布" and st.session_state.get("doc_topic_dist") is not None:
            topics = [f"主题{i+1}" for i in range(st.session_state.doc_topic_dist.shape[1])]
            df = pd.DataFrame(st.session_state.doc_topic_dist, columns=topics)
            df.insert(0, '文档', st.session_state.file_names[:len(df)])
            st.dataframe(df.head(10), use_container_width=True)
            
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
                st.dataframe(df, use_container_width=True)
                
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
            from modules.text_statistics import create_multi_doc_statistics
            
            raw_texts = st.session_state.get("raw_texts", [])
            texts = st.session_state.get("texts", [])
            file_names = st.session_state.get("file_names", [])
            
            if raw_texts and texts:
                all_stats = create_multi_doc_statistics(raw_texts, texts, file_names)
                csv_content = all_stats.export_comparison()
                
                df = pd.read_csv(BytesIO(csv_content.encode('utf-8-sig')))
                st.dataframe(df, use_container_width=True)
                
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
                st.dataframe(df.head(20), use_container_width=True)
                
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
        st.dataframe(df, use_container_width=True)
        
        st.download_button("📥 下载聚类结果CSV", dataframe_to_csv(df),
                         f"clustering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    elif classification_labels:
        st.markdown("**分类结果**")
        data = {"文档": list(classification_labels.keys()), "分类标签": list(classification_labels.values())}
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
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
        st.dataframe(df, use_container_width=True)
        
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
