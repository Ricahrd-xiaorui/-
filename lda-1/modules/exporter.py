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

def generate_report_html(file_names, texts, lda_model, topic_keywords, doc_topic_dist, 
                         coherence_score, perplexity, pyldavis_html=None):
    """生成HTML分析报告"""
    # 处理格式化值
    coherence_value = f"{coherence_score:.4f}" if coherence_score is not None else "N/A"
    perplexity_value = f"{perplexity:.4f}" if perplexity is not None else "N/A"
    
    # 创建HTML报告
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LDA主题模型分析报告</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .section {{
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }}
            .topic-keywords {{
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 10px;
            }}
            .keyword {{
                display: inline-block;
                background-color: #e0f7fa;
                padding: 3px 8px;
                margin: 3px;
                border-radius: 3px;
            }}
            .metric {{
                font-size: 18px;
                font-weight: bold;
                color: #0288d1;
            }}
            .pyldavis-container {{
                width: 100%;
                height: 800px;
                border: none;
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
        </style>
    </head>
    <body>
        <h1>LDA主题模型分析报告</h1>
        <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="section">
            <h2>1. 分析概述</h2>
            <p>本报告使用潜在狄利克雷分配(LDA)算法对文本集合进行主题建模分析。</p>
            
            <h3>分析数据</h3>
            <ul>
                <li>文档数量: {len(texts)}</li>
                <li>主题数量: {lda_model.num_topics}</li>
            </ul>
            
            <h3>模型评估指标</h3>
            <ul>
                <li>连贯性分数 (Coherence Score): <span class="metric">{coherence_value}</span> (越高越好)</li>
                <li>困惑度 (Perplexity): <span class="metric">{perplexity_value}</span></li>
            </ul>
        </div>
        
        <div class="section">
            <h2>2. 主题关键词</h2>
            <p>LDA模型识别出的{lda_model.num_topics}个主题及其关键词如下:</p>
    """
    
    # 添加主题关键词
    for topic_id, keywords in topic_keywords.items():
        html += f"""
            <div class="topic-keywords">
                <h3>主题 {topic_id + 1}</h3>
                <p>
        """
        for word in keywords[:20]:  # 显示前20个关键词
            html += f'<span class="keyword">{word}</span> '
        html += """
                </p>
            </div>
        """
    
    # 添加文档-主题分布
    html += """
        </div>
        
        <div class="section">
            <h2>3. 文档-主题分布</h2>
            <p>以下表格展示了每个文档在各个主题上的分布比例:</p>
            
            <table>
                <tr>
                    <th>文档</th>
    """
    
    # 添加表头
    for i in range(lda_model.num_topics):
        html += f"<th>主题 {i+1}</th>"
    html += "</tr>"
    
    # 添加表格内容
    for i, file_name in enumerate(file_names[:len(doc_topic_dist)]):
        html += f"<tr><td>{file_name}</td>"
        for j in range(doc_topic_dist.shape[1]):
            # 格式化数值
            html += f"<td>{doc_topic_dist[i, j]:.4f}</td>"
        html += "</tr>"
    
    html += """
            </table>
        </div>
    """
    
    # 如果有PyLDAvis可视化，添加到报告中
    if pyldavis_html:
        # 提取PyLDAvis的JS和HTML内容
        html += """
        <div class="section">
            <h2>4. 交互式主题可视化 (PyLDAvis)</h2>
            <p>以下是交互式主题模型可视化:</p>
            
            <iframe class="pyldavis-container" srcdoc='""" + pyldavis_html.replace("'", "\\'") + """'></iframe>
        </div>
        """
    
    # 结束HTML
    html += """
        <div class="section">
            <h2>5. 结论与建议</h2>
            <p>通过LDA主题模型分析，我们可以观察到文本集合中的主要主题分布。基于分析结果，可以进一步理解文档的内容结构，挖掘潜在的主题关联。</p>
            
            <p>建议:</p>
            <ul>
                <li>关注主题关键词，了解每个主题的核心内容</li>
                <li>分析文档-主题分布，识别文档的主要主题</li>
                <li>比较不同主题之间的关系，发现潜在的内容联系</li>
            </ul>
        </div>
        
        <footer>
            <p>生成自: 政策文件LDA主题模型可视化分析系统 | 版本 1.0.0</p>
        </footer>
    </body>
    </html>
    """
    
    return html

def dataframe_to_csv(df):
    """将DataFrame转换为CSV字符串"""
    return df.to_csv(index=False).encode('utf-8-sig')

def render_exporter():
    """渲染结果导出模块"""
    st.header("结果导出")
    
    # 功能介绍
    with st.expander("📖 功能介绍", expanded=False):
        st.markdown("""
        **结果导出模块** 支持将分析结果导出为多种格式，便于报告撰写和数据共享。
        
        **导出类型：**
        
        1. **📄 分析报告**
           - 生成完整的HTML格式分析报告
           - 包含模型概述、主题关键词、文档分布等内容
           - 可选择是否包含PyLDAvis交互式可视化
           - 支持浏览器打印为PDF
        
        2. **📊 数据表格**
           - **主题关键词**：导出每个主题的关键词列表（CSV格式）
           - **文档-主题分布**：导出每个文档在各主题上的概率分布
           - **主题相似度矩阵**：导出主题间的余弦相似度矩阵
        
        3. **💾 模型文件**
           - 导出训练好的LDA模型文件（ZIP压缩包）
           - 包含.gensim模型文件和.pkl状态文件
           - 可用于后续加载和继续分析
        
        **文件格式说明：**
        - **HTML**：网页格式，支持交互式可视化，可用浏览器打开
        - **CSV**：表格格式，可用Excel打开，支持中文（UTF-8-BOM编码）
        - **ZIP**：压缩包格式，包含模型相关的所有文件
        
        **使用建议：**
        - 撰写报告时建议导出HTML分析报告
        - 需要进一步数据分析时导出CSV表格
        - 需要保存模型供后续使用时导出模型文件
        """)
    
    # 检查是否完成了模型训练
    if not st.session_state.training_complete or not st.session_state.lda_model:
        st.warning('请先在"模型训练"选项卡中完成LDA模型训练')
        return
    
    # 创建选项卡
    export_tabs = st.tabs(["导出分析报告", "导出数据表格", "导出模型"])
    
    # 导出分析报告选项卡
    with export_tabs[0]:
        st.subheader("导出完整分析报告")
        
        # 报告格式选择
        report_format = st.radio(
            "选择报告格式",
            ["HTML", "PDF"],
            horizontal=True,
            help="HTML格式支持交互式可视化，PDF格式适合打印",
            key="report_format_radio"
        )
        
        # 报告内容选择
        st.write("选择要包含在报告中的内容:")
        include_topics = st.checkbox("主题关键词", value=True, key="include_topics_checkbox")
        include_doc_dist = st.checkbox("文档-主题分布", value=True, key="include_doc_dist_checkbox")
        include_pyldavis = st.checkbox("交互式PyLDAvis可视化", value=True, key="include_pyldavis_checkbox")
        include_wordcloud = st.checkbox("主题词云", value=True, key="include_wordcloud_checkbox")
        
        # 生成报告按钮
        if st.button("生成分析报告", key="generate_report"):
            with st.spinner(f"正在生成{report_format}分析报告..."):
                try:
                    # 准备数据
                    pyldavis_html = st.session_state.pyldavis_html if include_pyldavis else None
                    
                    # 生成HTML报告
                    html_report = generate_report_html(
                        st.session_state.file_names,
                        st.session_state.texts,
                        st.session_state.lda_model,
                        st.session_state.topic_keywords if include_topics else {},
                        st.session_state.doc_topic_dist if include_doc_dist else np.array([]),
                        st.session_state.coherence_score,
                        st.session_state.perplexity,
                        pyldavis_html
                    )
                    
                    # 根据选择的格式导出
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    if report_format == "HTML":
                        # 保存HTML报告
                        report_path = os.path.join("results", f"lda_report_{timestamp}.html")
                        os.makedirs(os.path.dirname(report_path), exist_ok=True)
                        
                        with open(report_path, "w", encoding="utf-8") as f:
                            f.write(html_report)
                        
                        # 提供下载链接
                        with open(report_path, "rb") as f:
                            report_data = f.read()
                        
                        st.download_button(
                            label="下载HTML报告",
                            data=report_data,
                            file_name=f"lda_report_{timestamp}.html",
                            mime="text/html"
                        )
                        
                        st.success(f"HTML报告已生成: {report_path}")
                        log_message(f"HTML报告已生成: {report_path}", level="success")
                    
                    elif report_format == "PDF":
                        # 暂时提供HTML版本，提示PDF功能待实现
                        st.warning("PDF导出功能正在开发中，目前将提供HTML版本")
                        
                        # 保存HTML报告
                        report_path = os.path.join("results", f"lda_report_{timestamp}.html")
                        os.makedirs(os.path.dirname(report_path), exist_ok=True)
                        
                        with open(report_path, "w", encoding="utf-8") as f:
                            f.write(html_report)
                        
                        # 提供下载链接
                        with open(report_path, "rb") as f:
                            report_data = f.read()
                        
                        st.download_button(
                            label="下载HTML报告",
                            data=report_data,
                            file_name=f"lda_report_{timestamp}.html",
                            mime="text/html"
                        )
                        
                        st.info("提示: 可以使用浏览器的打印功能将HTML转换为PDF")
                        log_message(f"HTML报告已生成: {report_path}", level="success")
                
                except Exception as e:
                    st.error(f"生成报告时出错: {str(e)}")
                    log_message(f"生成报告失败: {str(e)}", level="error")
    
    # 导出数据表格选项卡
    with export_tabs[1]:
        st.subheader("导出数据表格")
        
        # 数据表格选择
        data_type = st.selectbox(
            "选择要导出的数据",
            ["主题关键词", "文档-主题分布", "主题相似度矩阵"],
            key="export_data_type_select"
        )
        
        if data_type == "主题关键词":
            # 准备主题关键词数据
            if st.session_state.topic_keywords:
                # 创建一个包含所有主题关键词的DataFrame
                all_keywords = {}
                max_keywords = 0
                
                for topic_id, keywords in st.session_state.topic_keywords.items():
                    all_keywords[f"主题{topic_id+1}"] = keywords
                    max_keywords = max(max_keywords, len(keywords))
                
                # 确保所有列的长度相同
                for topic, keywords in all_keywords.items():
                    if len(keywords) < max_keywords:
                        all_keywords[topic] = keywords + [""] * (max_keywords - len(keywords))
                
                # 创建DataFrame
                df = pd.DataFrame(all_keywords)
                
                # 显示预览
                st.write("数据预览:")
                st.dataframe(df.head(10), use_container_width=True)
                
                # 提供下载按钮
                csv = dataframe_to_csv(df)
                st.download_button(
                    label="下载CSV",
                    data=csv,
                    file_name=f"topic_keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("没有可用的主题关键词数据")
        
        elif data_type == "文档-主题分布":
            # 准备文档-主题分布数据
            if st.session_state.doc_topic_dist is not None:
                # 创建DataFrame
                topics = [f"主题{i+1}" for i in range(st.session_state.doc_topic_dist.shape[1])]
                df = pd.DataFrame(st.session_state.doc_topic_dist, columns=topics)
                df['文档'] = st.session_state.file_names[:len(df)]
                
                # 调整列顺序，将文档列放在最前面
                cols = df.columns.tolist()
                cols = [cols[-1]] + cols[:-1]
                df = df[cols]
                
                # 显示预览
                st.write("数据预览:")
                st.dataframe(df.head(10), use_container_width=True)
                
                # 提供下载按钮
                csv = dataframe_to_csv(df)
                st.download_button(
                    label="下载CSV",
                    data=csv,
                    file_name=f"doc_topic_dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("没有可用的文档-主题分布数据")
        
        elif data_type == "主题相似度矩阵":
            # 计算主题相似度矩阵
            with st.spinner("正在计算主题相似度矩阵..."):
                try:
                    # 获取主题向量
                    num_topics = st.session_state.lda_model.num_topics
                    topic_vectors = []
                    
                    for i in range(num_topics):
                        topic_vector = [0] * len(st.session_state.dictionary)
                        for word_id, weight in st.session_state.lda_model.get_topic_terms(i, topn=len(st.session_state.dictionary)):
                            topic_vector[word_id] = weight
                        topic_vectors.append(topic_vector)
                    
                    topic_vectors = np.array(topic_vectors)
                    
                    # 计算余弦相似度
                    similarity_matrix = np.zeros((num_topics, num_topics))
                    for i in range(num_topics):
                        for j in range(num_topics):
                            # 避免自己与自己比较
                            if i == j:
                                similarity_matrix[i, j] = 1.0
                            else:
                                # 计算余弦相似度
                                dot_product = np.dot(topic_vectors[i], topic_vectors[j])
                                norm_i = np.linalg.norm(topic_vectors[i])
                                norm_j = np.linalg.norm(topic_vectors[j])
                                
                                if norm_i > 0 and norm_j > 0:
                                    similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
                                else:
                                    similarity_matrix[i, j] = 0
                    
                    # 创建DataFrame
                    topics = [f"主题{i+1}" for i in range(num_topics)]
                    df = pd.DataFrame(similarity_matrix, index=topics, columns=topics)
                    
                    # 显示预览
                    st.write("数据预览:")
                    st.dataframe(df, use_container_width=True)
                    
                    # 提供下载按钮
                    csv = dataframe_to_csv(df.reset_index().rename(columns={"index": "主题"}))
                    st.download_button(
                        label="下载CSV",
                        data=csv,
                        file_name=f"topic_similarity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"计算主题相似度矩阵时出错: {str(e)}")
                    log_message(f"计算主题相似度矩阵失败: {str(e)}", level="error")
    
    # 导出模型选项卡
    with export_tabs[2]:
        st.subheader("导出LDA模型")
        
        if st.session_state.model_path:
            st.write(f"当前模型已保存在: {st.session_state.model_path}")
            
            # 创建模型文件的ZIP包
            if st.button("导出模型文件", key="export_model"):
                with st.spinner("正在准备模型文件..."):
                    try:
                        # 创建临时目录
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # 准备要打包的文件
                            model_files = [
                                f"{st.session_state.model_path}.gensim",
                                f"{st.session_state.model_path}.pkl"
                            ]
                            
                            # 创建ZIP文件路径
                            zip_path = os.path.join(temp_dir, "lda_model.zip")
                            
                            # 创建ZIP文件
                            with zipfile.ZipFile(zip_path, 'w') as zipf:
                                for file in model_files:
                                    if os.path.exists(file):
                                        zipf.write(file, os.path.basename(file))
                            
                            # 读取ZIP文件
                            with open(zip_path, "rb") as f:
                                zip_data = f.read()
                            
                            # 提供下载按钮
                            st.download_button(
                                label="下载模型文件",
                                data=zip_data,
                                file_name=f"lda_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip"
                            )
                            
                            st.success("模型文件已准备好，请点击上方按钮下载")
                            log_message("模型文件已导出", level="success")
                    
                    except Exception as e:
                        st.error(f"导出模型时出错: {str(e)}")
                        log_message(f"导出模型失败: {str(e)}", level="error")
        else:
            st.warning("未找到保存的模型文件，请在模型训练完成后再尝试导出") 