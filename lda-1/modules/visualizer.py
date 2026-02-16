import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap
import networkx as nx
import pyLDAvis
import pyLDAvis.gensim_models
import tempfile
from pathlib import Path
import time
from datetime import datetime
import io
import base64
from utils.session_state import get_session_state, log_message, update_progress
from utils.font_config import setup_matplotlib_chinese, get_plotly_font, get_label, L

# 配置matplotlib中文字体
setup_matplotlib_chinese()

def get_system_font_path():
    """获取系统中文字体路径"""
    import platform
    import matplotlib.font_manager as fm
    
    # 优先检查项目fonts目录（用于Streamlit Cloud部署）
    project_font_paths = [
        "fonts/NotoSansSC-Regular.otf",
        "fonts/NotoSansCJKsc-Regular.otf",
        "fonts/SourceHanSansSC-Regular.otf",
        "fonts/simhei.ttf",
        "fonts/msyh.ttc",
    ]
    
    for font_path in project_font_paths:
        if os.path.exists(font_path):
            log_message(f"使用项目字体: {font_path}", level="info")
            return font_path
    
    system = platform.system()
    
    # 优先使用matplotlib检测到的中文字体
    chinese_font_names = []
    if system == "Windows":
        chinese_font_names = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
    elif system == "Darwin":  # macOS
        chinese_font_names = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        chinese_font_names = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 
                             'Droid Sans Fallback', 'AR PL UMing CN']
    
    # 从matplotlib字体管理器中查找
    for font in fm.fontManager.ttflist:
        if font.name in chinese_font_names:
            log_message(f"使用系统字体: {font.fname}", level="info")
            return font.fname
    
    # 备用：直接检查常见字体文件路径
    font_paths = []
    if system == "Windows":
        font_paths = [
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simsun.ttc",
            r"C:\Windows\Fonts\simkai.ttf",
        ]
    elif system == "Darwin":  # macOS
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    else:  # Linux
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
    
    # 检查字体文件是否存在
    for font_path in font_paths:
        if os.path.exists(font_path):
            log_message(f"使用系统字体: {font_path}", level="info")
            return font_path
    
    # 如果没有找到，返回None（WordCloud会使用默认字体）
    log_message("未找到中文字体，词云可能无法正确显示中文", level="warning")
    return None

class LDAVisualizer:
    """LDA主题模型可视化类"""
    
    def __init__(self, lda_model, corpus, dictionary, texts, doc_topic_dist=None, file_names=None):
        self.lda_model = lda_model
        self.corpus = corpus
        self.dictionary = dictionary
        self.texts = texts
        self.doc_topic_dist = doc_topic_dist
        self.file_names = file_names if file_names else [f"文档{i+1}" for i in range(len(corpus))]
    
    def generate_wordcloud(self, topic_id, max_words=50, width=800, height=400):
        """为指定主题生成词云"""
        # 获取主题词分布
        topic_words = dict(self.lda_model.show_topic(topic_id, topn=max_words))
        
        # 设置词云颜色
        colors = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        color = colors[topic_id % len(colors)]
        
        # 获取系统字体路径
        font_path = get_system_font_path()
        
        # 生成词云
        cloud = WordCloud(
            width=width,
            height=height,
            font_path=font_path,  # 使用动态检测的字体
            background_color='white',
            colormap='tab10',
            color_func=lambda *args, **kwargs: color,
            max_words=max_words,
            max_font_size=300,
            random_state=42
        )
        
        # 根据词频生成词云
        cloud.generate_from_frequencies(topic_words)
        
        return cloud
    
    def generate_doc_topic_heatmap(self, normalize=True):
        """生成文档-主题分布热图数据"""
        # 获取文档-主题分布
        if self.doc_topic_dist is None:
            doc_topic_dist = []
            for i, doc in enumerate(self.corpus):
                # 获取文档的主题分布
                topics = self.lda_model.get_document_topics(doc, minimum_probability=0.0)
                doc_topic_dist.append([prob for _, prob in sorted(topics)])
            
            doc_topic_dist = np.array(doc_topic_dist)
        else:
            doc_topic_dist = self.doc_topic_dist
            
        # 归一化
        if normalize and doc_topic_dist.shape[0] > 0:
            doc_topic_dist = doc_topic_dist / doc_topic_dist.sum(axis=1, keepdims=True)
        
        # 创建数据框
        topics = [f"主题{i+1}" for i in range(doc_topic_dist.shape[1])]
        
        # 使用实际文件名或默认名称
        df = pd.DataFrame(doc_topic_dist, columns=topics)
        df['文档'] = self.file_names[:len(df)]
        
        return df
    
    def generate_pyldavis(self):
        """生成PyLDAvis可视化"""
        try:
            # 准备PyLDAvis数据
            prepared_data = pyLDAvis.gensim_models.prepare(
                self.lda_model, self.corpus, self.dictionary, sort_topics=False
            )
            
            # 转换为HTML
            html_string = pyLDAvis.prepared_data_to_html(prepared_data)
            
            return html_string
        except Exception as e:
            log_message(f"生成PyLDAvis可视化失败: {str(e)}", level="error")
            return None
    
    def generate_topic_word_dist(self, num_words=20):
        """生成主题词分布数据"""
        topic_word_data = []
        
        for topic_id in range(self.lda_model.num_topics):
            # 获取主题词及概率
            topic_words = self.lda_model.show_topic(topic_id, topn=num_words)
            
            for word, prob in topic_words:
                topic_word_data.append({
                    '主题': f'主题{topic_id+1}',
                    '词语': word,
                    '概率': prob
                })
        
        return pd.DataFrame(topic_word_data)
    
    def generate_doc_clusters(self, method='tsne', n_clusters=None):
        """生成文档聚类可视化数据"""
        # 获取文档-主题分布
        if self.doc_topic_dist is None:
            doc_topic_dist = []
            for i, doc in enumerate(self.corpus):
                topics = self.lda_model.get_document_topics(doc, minimum_probability=0.0)
                doc_topic_dist.append([prob for _, prob in sorted(topics)])
            
            doc_topic_dist = np.array(doc_topic_dist)
        else:
            doc_topic_dist = self.doc_topic_dist
        
        # 如果未指定聚类数，使用主题数
        if n_clusters is None:
            n_clusters = self.lda_model.num_topics
        
        # 执行降维
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(doc_topic_dist)-1)))
            embedding = reducer.fit_transform(doc_topic_dist)
        elif method.lower() == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=min(15, max(2, len(doc_topic_dist)-1)))
            embedding = reducer.fit_transform(doc_topic_dist)
        
        # 执行聚类
        if len(doc_topic_dist) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(doc_topic_dist)
        else:
            # 如果文档数少于聚类数，直接使用文档的主要主题作为聚类
            clusters = np.argmax(doc_topic_dist, axis=1)
        
        # 获取每个文档的主导主题
        dominant_topics = np.argmax(doc_topic_dist, axis=1)
        
        # 创建数据框
        df = pd.DataFrame({
            'x': embedding[:, 0],
            'y': embedding[:, 1],
            '聚类': [f'聚类{i+1}' for i in clusters],
            '主导主题': [f'主题{i+1}' for i in dominant_topics],
            '文档': self.file_names[:len(doc_topic_dist)]
        })
        
        return df
    
    def generate_topic_similarity_network(self, threshold=0.2):
        """生成主题相似性网络"""
        num_topics = self.lda_model.num_topics
        
        # 计算主题之间的相似度矩阵
        topic_vectors = []
        for i in range(num_topics):
            topic_vector = [0] * len(self.dictionary)
            for word_id, weight in self.lda_model.get_topic_terms(i, topn=len(self.dictionary)):
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
        
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点
        for i in range(num_topics):
            G.add_node(i, name=f"主题{i+1}")
        
        # 添加边（只添加超过阈值的相似度）
        for i in range(num_topics):
            for j in range(i+1, num_topics):
                if similarity_matrix[i, j] >= threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
        
        return G, similarity_matrix

def render_visualizer():
    """渲染可视化分析模块"""
    st.header("可视化分析")
    
    # 功能介绍与操作手册
    with st.expander("📖 功能介绍与操作手册", expanded=False):
        st.markdown("""
        ## 📊 可视化分析模块
        
        **功能概述**：提供多种图表展示LDA主题模型的分析结果，帮助理解和解释主题结构。
        
        ---
        
        ### 🎯 使用场景
        
        | 可视化类型 | 适用场景 | 输出用途 |
        |------------|----------|----------|
        | 主题词云 | 快速了解主题内容 | 论文配图、报告展示 |
        | 文档-主题热图 | 分析文档归属 | 数据分析、分类依据 |
        | PyLDAvis | 深入探索主题结构 | 交互式分析、演示 |
        | 文档聚类 | 发现文档分布规律 | 聚类分析、异常检测 |
        | 主题词分布 | 比较主题差异 | 主题解释、论文配图 |
        | 相似性网络 | 分析主题关联 | 主题关系分析 |
        
        ---
        
        ### 📋 各可视化功能详解
        
        #### 1️⃣ 主题词云
        **功能**：以词云形式展示每个主题的关键词，词语大小表示重要程度。
        
        **操作步骤**：
        1. 选择要查看的主题编号
        2. 调整最大词数和词云宽度
        3. 点击"生成所有主题的词云"可一次性生成全部
        
        **参数说明**：
        - 最大词数：词云中显示的词语数量（10-100）
        - 词云宽度：图像宽度（400-1200像素）
        
        ---
        
        #### 2️⃣ 文档-主题分布热图
        **功能**：展示每个文档在各主题上的分布比例，颜色深浅表示关联强度。
        
        **解读方法**：
        - 颜色越深表示文档与该主题关联越强
        - 每行代表一个文档，每列代表一个主题
        - 可用于判断文档的主要主题归属
        
        ---
        
        #### 3️⃣ 交互式PyLDAvis
        **功能**：专业的LDA可视化工具，提供交互式主题探索。
        
        **界面说明**：
        - **左侧气泡图**：每个气泡代表一个主题
          - 气泡大小：主题在语料库中的占比
          - 气泡位置：主题间的相似度（距离越近越相似）
        - **右侧条形图**：选中主题的关键词
          - 蓝色条：词语在整个语料库中的频率
          - 红色条：词语在选中主题中的频率
        
        **交互操作**：
        - 点击气泡选择主题
        - 调整λ滑块改变词语排序方式
        - 悬停查看详细信息
        
        ---
        
        #### 4️⃣ 文档聚类
        **功能**：使用降维算法将文档映射到二维空间，展示文档分布。
        
        **降维算法**：
        - **t-SNE**：保持局部结构，适合发现聚类
        - **UMAP**：保持全局结构，计算更快
        
        **参数说明**：
        - 聚类数量：K-means聚类的簇数（建议与主题数相同）
        
        ---
        
        #### 5️⃣ 主题词分布
        **功能**：条形图展示每个主题的关键词概率分布。
        
        **参数说明**：
        - 每个主题显示词数：5-30个词
        
        ---
        
        #### 6️⃣ 主题相似性网络
        **功能**：网络图展示主题之间的相似关系。
        
        **参数说明**：
        - 相似度阈值：只显示相似度高于此值的连接（0.1-0.9）
        
        **解读方法**：
        - 节点代表主题
        - 连线表示主题间存在相似性
        - 节点颜色深浅表示连接数量
        
        ---
        
        ### 💡 使用建议
        
        **分析流程建议**：
        1. 先查看PyLDAvis获得整体印象
        2. 通过词云和词分布深入了解各主题内容
        3. 用文档聚类分析文档分布情况
        4. 用相似性网络分析主题关联
        
        **论文配图建议**：
        - 词云图：适合展示主题内容
        - 热图：适合展示文档分类结果
        - 聚类图：适合展示文档分布
        
        **图像保存**：
        - 每个可视化都提供保存按钮
        - HTML格式支持交互，适合演示
        - PNG格式适合论文配图
        
        ---
        
        ### ❓ 常见问题
        
        **Q: PyLDAvis加载很慢怎么办？**
        A: PyLDAvis需要计算大量数据，首次加载较慢，之后会使用缓存。
        
        **Q: 词云中文显示乱码怎么办？**
        A: 系统会自动检测中文字体，如仍有问题请确保系统安装了中文字体。
        
        **Q: 如何选择合适的聚类数量？**
        A: 建议与LDA主题数相同，或根据轮廓系数选择最优值。
        """)
    
    # 检查是否完成了模型训练
    if not st.session_state.training_complete or not st.session_state.lda_model:
        st.warning('请先在"模型训练"选项卡中完成LDA模型训练')
        return
    
    # 检查是否需要清理缓存（当模型更新时）
    current_model_id = id(st.session_state.lda_model)
    if 'last_model_id' not in st.session_state or st.session_state.last_model_id != current_model_id:
        # 清理可视化缓存
        st.session_state.pyldavis_html = None
        st.session_state.wordcloud_images = {}
        # 清理聚类缓存
        keys_to_remove = [k for k in st.session_state.keys() if k.startswith('tsne_') or k.startswith('umap_')]
        for key in keys_to_remove:
            del st.session_state[key]
        st.session_state.last_model_id = current_model_id
        log_message("检测到新模型，已清理可视化缓存", level="info")
    
    # 默认启用所有可视化选项
    for key in st.session_state.viz_options:
        st.session_state.viz_options[key] = True
    
    # 创建可视化器
    visualizer = LDAVisualizer(
        st.session_state.lda_model,
        st.session_state.corpus,
        st.session_state.dictionary,
        st.session_state.texts,
        st.session_state.doc_topic_dist,
        st.session_state.file_names
    )
    
    # 可视化选项
    st.subheader("可视化结果")
    
    # 创建选项卡
    viz_tabs = st.tabs([
        "主题词云", 
        "文档-主题分布", 
        "交互式PyLDAvis", 
        "文档聚类", 
        "主题词分布", 
        "主题相似性网络"
    ])
    
    # 主题词云选项卡
    with viz_tabs[0]:
        # 移除条件判断，直接显示内容
        st.subheader("主题词云")
        
        # 获取模型实际的主题数量
        actual_num_topics = st.session_state.lda_model.num_topics if st.session_state.lda_model else st.session_state.num_topics
        
        # 选择要查看的主题
        topic_id = st.selectbox(
            "选择主题",
            range(actual_num_topics),
            format_func=lambda x: f"主题 {x+1}",
            key="wordcloud_topic_select"
        )
        
        # 词云参数
        col1, col2 = st.columns(2)
        with col1:
            max_words = st.slider("最大词数", 10, 100, 50, key="wordcloud_max_words")
        with col2:
            width = st.slider("词云宽度", 400, 1200, 800, key="wordcloud_width")
        
        # 生成并显示词云
        with st.spinner("正在生成词云..."):
            # 检查缓存
            cache_key = f"wordcloud_{topic_id}_{max_words}_{width}"
            if cache_key not in st.session_state.wordcloud_images:
                # 生成词云
                wordcloud = visualizer.generate_wordcloud(
                    topic_id=topic_id,
                    max_words=max_words,
                    width=width,
                    height=400
                )
                
                # 将词云转换为图像
                fig, ax = plt.subplots(figsize=(width/100, 400/100))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                
                # 缓存图像
                st.session_state.wordcloud_images[cache_key] = fig
            
            # 显示词云
            st.pyplot(st.session_state.wordcloud_images[cache_key])
        
        # 生成所有主题的词云按钮
        if st.button("生成所有主题的词云", key="gen_all_wordclouds"):
            with st.spinner("正在生成所有主题的词云..."):
                # 创建图表（使用实际主题数）
                n_topics = actual_num_topics
                n_cols = min(3, n_topics)
                n_rows = (n_topics + n_cols - 1) // n_cols  # 向上取整
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                
                # 扁平化轴数组以便索引
                if n_rows > 1 and n_cols > 1:
                    axes = axes.flatten()
                elif n_rows == 1 and n_cols > 1:
                    axes = axes
                elif n_rows > 1 and n_cols == 1:
                    axes = [axes[i] for i in range(n_rows)]
                else:
                    axes = [axes]
                
                # 为每个主题生成词云
                for i in range(n_topics):
                    # 生成词云
                    wordcloud = visualizer.generate_wordcloud(
                        topic_id=i,
                        max_words=30,
                        width=400,
                        height=300
                    )
                    
                    # 在子图中显示
                    axes[i].imshow(wordcloud, interpolation='bilinear')
                    axes[i].set_title(f'主题 {i+1}')
                    axes[i].axis("off")
                
                # 隐藏空白子图
                for i in range(n_topics, len(axes)):
                    axes[i].axis("off")
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 保存图像
                save_path = os.path.join("results", f"topic_wordclouds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                st.success(f"已保存词云图像至: {save_path}")
    
    # 文档-主题分布选项卡
    with viz_tabs[1]:
        # 移除条件判断，直接显示内容
        st.subheader("文档-主题分布")
        
        with st.spinner("正在生成文档-主题分布热图..."):
            # 生成热图数据
            df = visualizer.generate_doc_topic_heatmap()
            
            # 准备热图数据
            df_melt = df.melt(id_vars='文档', var_name='主题', value_name='权重')
            
            # 绘制热图
            fig = px.density_heatmap(
                df_melt, 
                x='主题', 
                y='文档',
                z='权重',
                color_continuous_scale='Viridis',
                title='文档-主题分布热图'
            )
            
            # 调整图表布局
            fig.update_layout(
                width=800,
                height=max(400, len(df) * 30),  # 根据文档数量调整高度
                xaxis_title='主题',
                yaxis_title='文档',
                coloraxis_colorbar=dict(title='权重')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 保存热图按钮
            if st.button("保存热图", key="save_heatmap"):
                # 保存图像
                save_path = os.path.join("results", f"doc_topic_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.write_html(save_path)
                st.success(f"已保存热图至: {save_path}")
        
        # 显示原始数据表格
        with st.expander("查看原始数据"):
            st.dataframe(df, use_container_width=True)
            
            # 下载CSV按钮
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="下载CSV",
                data=csv,
                file_name=f"doc_topic_dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # 交互式PyLDAvis选项卡
    with viz_tabs[2]:
        # 移除条件判断，直接显示内容
        st.subheader("交互式PyLDAvis可视化")
        
        # 检查缓存
        if st.session_state.pyldavis_html is None:
            with st.spinner("正在生成PyLDAvis可视化..."):
                # 生成PyLDAvis
                html_string = visualizer.generate_pyldavis()
                
                if html_string:
                    # 缓存HTML
                    st.session_state.pyldavis_html = html_string
                else:
                    st.error("生成PyLDAvis可视化失败")
        
        # 显示PyLDAvis
        if st.session_state.pyldavis_html:
            st.components.v1.html(st.session_state.pyldavis_html, width=1000, height=800)
            
            # 保存PyLDAvis按钮
            if st.button("保存PyLDAvis", key="save_pyldavis"):
                # 保存HTML
                save_path = os.path.join("results", f"pyldavis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(st.session_state.pyldavis_html)
                
                st.success(f"已保存PyLDAvis可视化至: {save_path}")
    
    # 文档聚类选项卡
    with viz_tabs[3]:
        # 移除条件判断，直接显示内容
        st.subheader("文档聚类可视化")
        
        # 获取模型实际的主题数量
        actual_num_topics_for_cluster = st.session_state.lda_model.num_topics if st.session_state.lda_model else st.session_state.num_topics
        
        col1, col2 = st.columns(2)
        with col1:
            method = st.radio("降维方法", ["t-SNE", "UMAP"], horizontal=True, key="clustering_method_radio")
        with col2:
            n_clusters = st.slider(
                "聚类数量", 
                min_value=2, 
                max_value=min(10, len(st.session_state.corpus)),
                value=min(actual_num_topics_for_cluster, len(st.session_state.corpus)),
                key="clustering_n_clusters"
            )
        
        with st.spinner(f"正在使用{method}进行文档聚类..."):
            # 生成聚类数据
            df = visualizer.generate_doc_clusters(
                method=method.lower().replace('-', ''), 
                n_clusters=n_clusters
            )
            
            # 缓存键
            cache_key = f"{method.lower()}_{n_clusters}"
            
            # 检查缓存
            if cache_key not in st.session_state:
                st.session_state[cache_key] = df
            
            # 使用缓存的数据
            df = st.session_state[cache_key]
            
            # 绘制散点图
            fig = px.scatter(
                df, 
                x='x', 
                y='y',
                color='聚类',
                symbol='主导主题',
                hover_data=['文档'],
                title=f'文档聚类可视化 ({method})',
                labels={'x': '', 'y': ''},
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            
            # 调整图表布局
            fig.update_layout(
                width=800,
                height=600,
                legend_title_text='聚类和主题'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 保存图表按钮
            if st.button("保存聚类图", key="save_clusters"):
                # 保存图像
                save_path = os.path.join("results", f"doc_clusters_{method.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.write_html(save_path)
                st.success(f"已保存聚类图至: {save_path}")
            
            # 显示聚类详情
            with st.expander("查看聚类详情"):
                st.dataframe(df, use_container_width=True)
    
    # 主题词分布选项卡
    with viz_tabs[4]:
        # 移除条件判断，直接显示内容
        st.subheader("主题词分布")
        
        # 获取模型实际的主题数量
        actual_num_topics_for_dist = st.session_state.lda_model.num_topics if st.session_state.lda_model else st.session_state.num_topics
        
        # 参数设置
        num_words = st.slider("每个主题显示词数", 5, 30, 15, key="topic_word_num_words")
        
        with st.spinner("正在生成主题词分布..."):
            # 生成主题词分布数据
            df = visualizer.generate_topic_word_dist(num_words=num_words)
            
            # 绘制条形图
            fig = px.bar(
                df, 
                x='概率', 
                y='词语',
                color='主题',
                facet_col='主题',
                facet_col_wrap=2,  # 每行显示2个主题
                orientation='h',
                title='主题词分布',
                labels={'概率': '词语概率', '词语': '词语'},
                height=max(600, num_words * 30 * (actual_num_topics_for_dist + 1) // 2)  # 根据词数和主题数调整高度
            )
            
            # 调整图表布局
            fig.update_layout(
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            # 对每个子图进行调整
            for i in range(actual_num_topics_for_dist):
                fig.update_yaxes(showticklabels=True, col=i+1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 保存图表按钮
            if st.button("保存主题词分布图", key="save_topic_word_dist"):
                # 保存图像
                save_path = os.path.join("results", f"topic_word_dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.write_html(save_path)
                st.success(f"已保存主题词分布图至: {save_path}")
    
    # 主题相似性网络选项卡
    with viz_tabs[5]:
        # 移除条件判断，直接显示内容
        st.subheader("主题相似性网络")
        
        # 参数设置
        threshold = st.slider("相似度阈值", 0.1, 0.9, 0.3, 0.05, key="similarity_threshold")
        
        with st.spinner("正在生成主题相似性网络..."):
            # 生成主题相似性网络
            G, similarity_matrix = visualizer.generate_topic_similarity_network(threshold=threshold)
            
            # 显示网络图
            if G.number_of_edges() > 0:
                # 使用NetworkX和Plotly生成交互式网络图
                pos = nx.spring_layout(G, seed=42)
                
                # 准备边和节点数据
                edge_x = []
                edge_y = []
                edge_weights = []
                
                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_weights.append(edge[2]['weight'])
                
                # 创建边迹
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color='rgba(150,150,150,0.7)'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                # 节点数据
                node_x = []
                node_y = []
                node_text = []
                node_adjacencies = []
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(f"主题{node+1}")
                    node_adjacencies.append(len(list(G.neighbors(node))))
                
                # 创建节点迹
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition="top center",
                    marker=dict(
                        showscale=True,
                        colorscale='YlGnBu',
                        reversescale=True,
                        color=node_adjacencies,
                        size=15,
                        colorbar=dict(
                            thickness=15,
                            title=dict(
                                text='连接数',
                                side='right'
                            ),
                            xanchor='left'
                        ),
                        line_width=2
                    )
                )
                
                # 创建网络图
                fig = go.Figure(data=[edge_trace, node_trace],
                                 layout=go.Layout(
                                     title=dict(
                                         text='主题相似性网络',
                                         font=dict(size=16)
                                     ),
                                     showlegend=False,
                                     hovermode='closest',
                                     margin=dict(b=20,l=5,r=5,t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                                 ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 保存网络图按钮
                if st.button("保存主题相似性网络", key="save_topic_network"):
                    # 保存图像
                    save_path = os.path.join("results", f"topic_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    fig.write_html(save_path)
                    st.success(f"已保存主题相似性网络图至: {save_path}")
                
                # 显示相似度矩阵
                with st.expander("查看主题相似度矩阵"):
                    # 创建相似度矩阵数据框
                    topics = [f"主题{i+1}" for i in range(len(similarity_matrix))]
                    sim_df = pd.DataFrame(similarity_matrix, index=topics, columns=topics)
                    
                    # 显示热图
                    fig = px.imshow(
                        sim_df,
                        text_auto='.2f',
                        color_continuous_scale='Viridis',
                        title='主题相似度矩阵'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"在阈值 {threshold} 下没有主题之间的连接。请尝试降低阈值。") 