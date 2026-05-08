# -*- coding: utf-8 -*-
"""
文本比较分析模块 (Comparative Analysis Module)
==============================================

本模块提供文本比较分析功能，包括：
- 文档相似度计算
- 共同关键词与差异关键词识别
- 相似段落检测
- 比较结果可视化
- 结果导出

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """比较结果数据类"""
    doc1_name: str
    doc2_name: str
    similarity: float
    common_keywords: List[str]
    doc1_unique_keywords: List[str]
    doc2_unique_keywords: List[str]
    similar_segments: List[Tuple[str, str, float]]


class ComparativeAnalyzer:
    """
    比较分析器 - 对比分析不同文本的异同
    
    Attributes:
        texts: 分词后的文本列表（每个文本是词语列表）
        file_names: 文档名称列表
        raw_texts: 原始文本列表（用于段落比较）
    
    Requirements: 5.2, 5.3, 5.6
    """
    
    def __init__(self, texts: List[List[str]], file_names: List[str], 
                 raw_texts: Optional[List[str]] = None):
        """
        初始化比较分析器
        
        Args:
            texts: 分词后的文本列表
            file_names: 文档名称列表
            raw_texts: 原始文本列表（可选，用于段落比较）
        """
        self.texts = texts if texts else []
        self.file_names = file_names if file_names else []
        self.raw_texts = raw_texts if raw_texts else []
        self._tfidf_matrix: Optional[np.ndarray] = None
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._similarity_matrix: Optional[np.ndarray] = None
    
    def _ensure_tfidf_matrix(self) -> None:
        """确保TF-IDF矩阵已计算"""
        if self._tfidf_matrix is None:
            if not self.texts:
                self._tfidf_matrix = np.array([])
                return
            
            # 将词语列表转换为空格分隔的字符串
            text_strings = [' '.join(words) for words in self.texts]
            
            self._vectorizer = TfidfVectorizer(max_features=1000)
            self._tfidf_matrix = self._vectorizer.fit_transform(text_strings).toarray()
    
    def calculate_similarity(self, doc1_idx: int, doc2_idx: int, 
                           method: str = 'cosine') -> float:
        """
        计算两个文档之间的相似度
        
        Args:
            doc1_idx: 第一个文档的索引
            doc2_idx: 第二个文档的索引
            method: 相似度计算方法，支持 'cosine'（余弦相似度）和 'jaccard'（Jaccard相似度）
        
        Returns:
            float: 相似度得分，范围 [0, 1]
        
        Requirements: 5.2
        """
        if not self.texts:
            return 0.0
        
        # 边界检查
        if doc1_idx < 0 or doc1_idx >= len(self.texts):
            return 0.0
        if doc2_idx < 0 or doc2_idx >= len(self.texts):
            return 0.0
        
        # 相同文档相似度为1
        if doc1_idx == doc2_idx:
            return 1.0
        
        if method == 'cosine':
            self._ensure_tfidf_matrix()
            if self._tfidf_matrix is None or len(self._tfidf_matrix) == 0:
                return 0.0
            
            vec1 = self._tfidf_matrix[doc1_idx].reshape(1, -1)
            vec2 = self._tfidf_matrix[doc2_idx].reshape(1, -1)
            
            similarity = cosine_similarity(vec1, vec2)[0][0]
            return float(similarity)
        
        elif method == 'jaccard':
            # Jaccard相似度：交集/并集
            set1 = set(self.texts[doc1_idx])
            set2 = set(self.texts[doc2_idx])
            
            if not set1 and not set2:
                return 1.0  # 两个空集视为相同
            if not set1 or not set2:
                return 0.0
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            return intersection / union if union > 0 else 0.0
        
        return 0.0
    
    def calculate_similarity_matrix(self, method: str = 'cosine') -> np.ndarray:
        """
        计算所有文档之间的相似度矩阵
        
        Args:
            method: 相似度计算方法
        
        Returns:
            np.ndarray: 相似度矩阵，形状为 (n_docs, n_docs)
        
        Requirements: 5.2
        """
        n_docs = len(self.texts)
        
        if n_docs == 0:
            return np.array([])
        
        if method == 'cosine':
            self._ensure_tfidf_matrix()
            if self._tfidf_matrix is None or len(self._tfidf_matrix) == 0:
                return np.array([])
            
            self._similarity_matrix = cosine_similarity(self._tfidf_matrix)
            return self._similarity_matrix
        
        elif method == 'jaccard':
            matrix = np.zeros((n_docs, n_docs))
            for i in range(n_docs):
                for j in range(n_docs):
                    matrix[i][j] = self.calculate_similarity(i, j, method='jaccard')
            self._similarity_matrix = matrix
            return matrix
        
        return np.array([])
    
    def find_common_keywords(self, doc_indices: List[int], top_n: int = 20) -> List[str]:
        """
        识别多个文档间的共同关键词
        
        Args:
            doc_indices: 要比较的文档索引列表
            top_n: 返回的关键词数量
        
        Returns:
            List[str]: 共同关键词列表
        
        Requirements: 5.3
        """
        if not doc_indices or not self.texts:
            return []
        
        # 验证索引有效性
        valid_indices = [i for i in doc_indices if 0 <= i < len(self.texts)]
        if len(valid_indices) < 2:
            return []
        
        # 获取每个文档的词集合
        word_sets = [set(self.texts[i]) for i in valid_indices]
        
        # 计算交集
        common_words = word_sets[0]
        for word_set in word_sets[1:]:
            common_words = common_words & word_set
        
        if not common_words:
            return []
        
        # 按总频率排序
        word_freq = Counter()
        for idx in valid_indices:
            for word in self.texts[idx]:
                if word in common_words:
                    word_freq[word] += 1
        
        return [word for word, _ in word_freq.most_common(top_n)]
    
    def find_unique_keywords(self, doc_idx: int, other_indices: List[int], 
                            top_n: int = 20) -> List[str]:
        """
        识别文档相对于其他文档的独特关键词
        
        Args:
            doc_idx: 目标文档索引
            other_indices: 其他文档索引列表
            top_n: 返回的关键词数量
        
        Returns:
            List[str]: 独特关键词列表
        
        Requirements: 5.3
        """
        if not self.texts:
            return []
        
        # 验证索引有效性
        if doc_idx < 0 or doc_idx >= len(self.texts):
            return []
        
        valid_other_indices = [i for i in other_indices if 0 <= i < len(self.texts) and i != doc_idx]
        
        # 获取目标文档的词集合
        target_words = set(self.texts[doc_idx])
        
        # 获取其他文档的词集合并集
        other_words = set()
        for idx in valid_other_indices:
            other_words.update(self.texts[idx])
        
        # 计算差集
        unique_words = target_words - other_words
        
        if not unique_words:
            return []
        
        # 按频率排序
        word_freq = Counter()
        for word in self.texts[doc_idx]:
            if word in unique_words:
                word_freq[word] += 1
        
        return [word for word, _ in word_freq.most_common(top_n)]
    
    def find_similar_segments(self, doc1_idx: int, doc2_idx: int, 
                             threshold: float = 0.5,
                             segment_size: int = 50) -> List[Tuple[str, str, float]]:
        """
        查找两个文档之间的相似段落
        
        Args:
            doc1_idx: 第一个文档的索引
            doc2_idx: 第二个文档的索引
            threshold: 相似度阈值
            segment_size: 段落大小（字符数）
        
        Returns:
            List[Tuple[str, str, float]]: (文档1段落, 文档2段落, 相似度) 列表
        
        Requirements: 5.6
        """
        if not self.raw_texts:
            return []
        
        # 验证索引有效性
        if doc1_idx < 0 or doc1_idx >= len(self.raw_texts):
            return []
        if doc2_idx < 0 or doc2_idx >= len(self.raw_texts):
            return []
        
        text1 = self.raw_texts[doc1_idx]
        text2 = self.raw_texts[doc2_idx]
        
        # 分割成段落
        segments1 = self._split_into_segments(text1, segment_size)
        segments2 = self._split_into_segments(text2, segment_size)
        
        if not segments1 or not segments2:
            return []
        
        # 计算段落间的相似度
        similar_pairs = []
        
        # 使用TF-IDF计算段落相似度
        all_segments = segments1 + segments2
        if len(all_segments) < 2:
            return []
        
        try:
            vectorizer = TfidfVectorizer(max_features=500)
            tfidf_matrix = vectorizer.fit_transform(all_segments).toarray()
            
            n1 = len(segments1)
            
            for i, seg1 in enumerate(segments1):
                for j, seg2 in enumerate(segments2):
                    vec1 = tfidf_matrix[i].reshape(1, -1)
                    vec2 = tfidf_matrix[n1 + j].reshape(1, -1)
                    
                    sim = cosine_similarity(vec1, vec2)[0][0]
                    
                    if sim >= threshold:
                        similar_pairs.append((seg1, seg2, float(sim)))
        except Exception:
            return []
        
        # 按相似度降序排序
        similar_pairs.sort(key=lambda x: -x[2])
        
        return similar_pairs[:20]  # 最多返回20对
    
    def _split_into_segments(self, text: str, segment_size: int) -> List[str]:
        """
        将文本分割成段落
        
        Args:
            text: 原始文本
            segment_size: 段落大小
        
        Returns:
            List[str]: 段落列表
        """
        if not text:
            return []
        
        # 首先按换行符分割
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        segments = []
        for para in paragraphs:
            if len(para) <= segment_size:
                if para:
                    segments.append(para)
            else:
                # 按句子分割长段落
                sentences = self._split_into_sentences(para)
                current_segment = ""
                
                for sentence in sentences:
                    if len(current_segment) + len(sentence) <= segment_size:
                        current_segment += sentence
                    else:
                        if current_segment:
                            segments.append(current_segment)
                        current_segment = sentence
                
                if current_segment:
                    segments.append(current_segment)
        
        return segments
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割成句子
        
        Args:
            text: 文本
        
        Returns:
            List[str]: 句子列表
        """
        import re
        # 中文句子分割
        sentences = re.split(r'([。！？；\n])', text)
        
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else '')
            if sentence.strip():
                result.append(sentence.strip())
        
        # 处理最后一个元素
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return result
    
    def compare_documents(self, doc1_idx: int, doc2_idx: int, 
                         top_n: int = 20) -> ComparisonResult:
        """
        全面比较两个文档
        
        Args:
            doc1_idx: 第一个文档的索引
            doc2_idx: 第二个文档的索引
            top_n: 返回的关键词数量
        
        Returns:
            ComparisonResult: 比较结果
        """
        doc1_name = self.file_names[doc1_idx] if doc1_idx < len(self.file_names) else f"文档{doc1_idx}"
        doc2_name = self.file_names[doc2_idx] if doc2_idx < len(self.file_names) else f"文档{doc2_idx}"
        
        # 计算相似度
        similarity = self.calculate_similarity(doc1_idx, doc2_idx)
        
        # 查找共同关键词
        common_keywords = self.find_common_keywords([doc1_idx, doc2_idx], top_n)
        
        # 查找各自独特关键词
        doc1_unique = self.find_unique_keywords(doc1_idx, [doc2_idx], top_n)
        doc2_unique = self.find_unique_keywords(doc2_idx, [doc1_idx], top_n)
        
        # 查找相似段落
        similar_segments = self.find_similar_segments(doc1_idx, doc2_idx)
        
        return ComparisonResult(
            doc1_name=doc1_name,
            doc2_name=doc2_name,
            similarity=similarity,
            common_keywords=common_keywords,
            doc1_unique_keywords=doc1_unique,
            doc2_unique_keywords=doc2_unique,
            similar_segments=similar_segments
        )
    
    def get_most_similar_pairs(self, top_n: int = 10) -> List[Tuple[str, str, float]]:
        """
        获取最相似的文档对
        
        Args:
            top_n: 返回的文档对数量
        
        Returns:
            List[Tuple[str, str, float]]: (文档1名, 文档2名, 相似度) 列表
        """
        if len(self.texts) < 2:
            return []
        
        # 计算相似度矩阵
        sim_matrix = self.calculate_similarity_matrix()
        
        if sim_matrix is None or len(sim_matrix) == 0:
            return []
        
        # 收集所有文档对的相似度
        pairs = []
        n_docs = len(self.texts)
        
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                pairs.append((
                    self.file_names[i] if i < len(self.file_names) else f"文档{i}",
                    self.file_names[j] if j < len(self.file_names) else f"文档{j}",
                    float(sim_matrix[i][j])
                ))
        
        # 按相似度降序排序
        pairs.sort(key=lambda x: -x[2])
        
        return pairs[:top_n]
    
    def export_comparison(self, doc_indices: Optional[List[int]] = None) -> str:
        """
        导出比较分析结果为CSV格式
        
        Args:
            doc_indices: 要导出的文档索引列表，如果为None则导出所有
        
        Returns:
            str: CSV格式字符串
        
        Requirements: 5.7
        """
        if not self.texts:
            return ""
        
        if doc_indices is None:
            doc_indices = list(range(len(self.texts)))
        
        # 计算相似度矩阵
        sim_matrix = self.calculate_similarity_matrix()
        
        if sim_matrix is None or len(sim_matrix) == 0:
            return ""
        
        # 构建数据
        data = []
        for i in doc_indices:
            for j in doc_indices:
                if i < j:
                    doc1_name = self.file_names[i] if i < len(self.file_names) else f"文档{i}"
                    doc2_name = self.file_names[j] if j < len(self.file_names) else f"文档{j}"
                    
                    common_kw = self.find_common_keywords([i, j], 10)
                    
                    data.append({
                        "文档1": doc1_name,
                        "文档2": doc2_name,
                        "相似度": round(float(sim_matrix[i][j]), 4),
                        "共同关键词": ", ".join(common_kw[:5])
                    })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='utf-8-sig')
    
    def export_similarity_matrix(self) -> str:
        """
        导出相似度矩阵为CSV格式
        
        Returns:
            str: CSV格式字符串
        """
        sim_matrix = self.calculate_similarity_matrix()
        
        if sim_matrix is None or len(sim_matrix) == 0:
            return ""
        
        # 使用文档名作为行列标签
        labels = [self.file_names[i] if i < len(self.file_names) else f"文档{i}" 
                  for i in range(len(self.texts))]
        
        df = pd.DataFrame(sim_matrix, index=labels, columns=labels)
        return df.to_csv(encoding='utf-8-sig')



# ============================================================================
# Streamlit UI 渲染函数
# ============================================================================

def render_comparative_analyzer():
    """
    渲染文本比较分析模块UI
    
    Requirements: 5.1, 5.4, 5.5, 5.7
    """
    import streamlit as st
    from utils.session_state import log_message
    
    st.header("🔍 文本比较分析")
    
    # 功能介绍与操作手册
    with st.expander("📖 功能介绍与操作手册", expanded=False):
        st.markdown("""
        ## 🔍 文本比较分析模块
        
        **功能概述**：对比分析不同政策文本的异同，支持相似度计算、关键词对比和相似段落检测。
        
        ---
        
        ### 🎯 使用场景
        
        | 场景 | 关注点 | 应用 |
        |------|--------|------|
        | 政策比较研究 | 相似度矩阵 | 发现相似政策文件 |
        | 差异分析 | 独特关键词 | 识别政策差异点 |
        | 共性分析 | 共同关键词 | 发现政策共同主题 |
        | 文本溯源 | 相似段落 | 追踪政策文本来源 |
        
        ---
        
        ### 📋 操作步骤
        
        **1. 文档选择**：
        - 选择要比较的两个或多个文档
        - 可以选择全部文档进行整体分析
        
        **2. 相似度分析**：
        - 查看文档间的相似度得分
        - 相似度热图展示整体关系
        
        **3. 关键词对比**：
        - 查看共同关键词（韦恩图）
        - 查看各文档独特关键词
        
        **4. 并排对比**：
        - 选择两个文档进行详细对比
        - 查看相似段落高亮显示
        
        **5. 导出结果**：
        - 导出相似度矩阵
        - 导出比较分析报告
        
        ---
        
        ### ⚙️ 相似度计算方法
        
        | 方法 | 原理 | 适用场景 |
        |------|------|----------|
        | 余弦相似度 | 基于TF-IDF向量夹角 | 通用，推荐使用 |
        | Jaccard相似度 | 基于词集合交并比 | 关注词汇重叠 |
        
        ---
        
        ### 💡 使用建议
        
        **学术研究建议**：
        - 相似度>0.8：高度相似，可能存在引用关系
        - 相似度0.5-0.8：中度相似，主题相近
        - 相似度<0.5：差异较大，主题不同
        
        **可视化建议**：
        - 热图适合展示多文档整体关系
        - 韦恩图适合展示2-3个文档的关键词重叠
        """)
    
    # 检查数据
    if not st.session_state.get("texts"):
        st.warning("⚠️ 请先在「文本预处理」标签页中完成文本预处理")
        return
    
    texts = st.session_state["texts"]
    file_names = st.session_state.get("file_names", [])
    raw_texts = st.session_state.get("raw_texts", [])
    
    if len(texts) < 2:
        st.warning("⚠️ 至少需要2个文档才能进行比较分析")
        return
    
    # 获取或创建分析器
    if "comparative_analyzer" not in st.session_state or st.session_state["comparative_analyzer"] is None:
        st.session_state["comparative_analyzer"] = ComparativeAnalyzer(texts, file_names, raw_texts)
    
    analyzer = st.session_state["comparative_analyzer"]
    
    # 创建标签页
    tabs = st.tabs([
        "📊 相似度矩阵",
        "🔄 文档对比",
        "🔑 关键词分析",
        "📝 相似段落",
        "💾 导出"
    ])
    
    # ========== 相似度矩阵 ==========
    with tabs[0]:
        st.subheader("文档相似度矩阵")
        
        # 参数设置
        col1, col2 = st.columns(2)
        with col1:
            similarity_method = st.selectbox(
                "相似度计算方法",
                ["余弦相似度 (cosine)", "Jaccard相似度 (jaccard)"],
                index=0,
                help="余弦相似度基于TF-IDF向量，Jaccard相似度基于词集合"
            )
        
        method = "cosine" if "cosine" in similarity_method else "jaccard"
        
        if st.button("📊 计算相似度矩阵", type="primary", key="calc_sim_matrix"):
            with st.spinner("正在计算相似度矩阵..."):
                sim_matrix = analyzer.calculate_similarity_matrix(method=method)
                st.session_state["similarity_matrix"] = sim_matrix
                st.session_state["similarity_method"] = method
                log_message(f"计算了 {len(texts)} 个文档的相似度矩阵")
        
        # 显示相似度矩阵
        if st.session_state.get("similarity_matrix") is not None:
            sim_matrix = st.session_state["similarity_matrix"]
            
            if len(sim_matrix) > 0:
                # 创建DataFrame用于显示
                labels = [file_names[i] if i < len(file_names) else f"文档{i}" 
                         for i in range(len(sim_matrix))]
                
                # 截断长文件名
                short_labels = [name[:15] + "..." if len(name) > 15 else name for name in labels]
                
                df = pd.DataFrame(sim_matrix, index=short_labels, columns=short_labels)
                
                # 使用Plotly绘制热图
                try:
                    import plotly.express as px
                    import plotly.graph_objects as go
                    
                    fig = px.imshow(
                        sim_matrix,
                        x=short_labels,
                        y=short_labels,
                        color_continuous_scale="RdYlBu_r",
                        aspect="auto",
                        title="文档相似度热图"
                    )
                    fig.update_layout(
                        xaxis_title="文档",
                        yaxis_title="文档",
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, width='stretch')
                    
                except ImportError:
                    st.dataframe(df.style.background_gradient(cmap='RdYlBu_r'), 
                               width='stretch')
                
                # 显示最相似的文档对
                st.markdown("---")
                st.subheader("最相似的文档对")
                
                similar_pairs = analyzer.get_most_similar_pairs(top_n=10)
                
                if similar_pairs:
                    pairs_df = pd.DataFrame(similar_pairs, columns=["文档1", "文档2", "相似度"])
                    pairs_df["相似度"] = pairs_df["相似度"].apply(lambda x: f"{x:.4f}")
                    st.dataframe(pairs_df, width='stretch', hide_index=True)
    
    # ========== 文档对比 ==========
    with tabs[1]:
        st.subheader("文档对比分析")
        
        # 选择要比较的文档
        col1, col2 = st.columns(2)
        
        with col1:
            doc1_name = st.selectbox(
                "选择第一个文档",
                file_names,
                index=0,
                key="compare_doc1"
            )
        
        with col2:
            # 默认选择第二个文档
            default_idx = 1 if len(file_names) > 1 else 0
            doc2_name = st.selectbox(
                "选择第二个文档",
                file_names,
                index=default_idx,
                key="compare_doc2"
            )
        
        if doc1_name == doc2_name:
            st.warning("请选择两个不同的文档进行比较")
        else:
            doc1_idx = file_names.index(doc1_name)
            doc2_idx = file_names.index(doc2_name)
            
            if st.button("🔄 开始对比", type="primary", key="start_compare"):
                with st.spinner("正在分析文档差异..."):
                    result = analyzer.compare_documents(doc1_idx, doc2_idx)
                    st.session_state["comparison_result"] = result
                    log_message(f"对比了文档: {doc1_name} vs {doc2_name}")
            
            # 显示对比结果
            if st.session_state.get("comparison_result"):
                result = st.session_state["comparison_result"]
                
                # 相似度指标
                st.markdown("### 📊 相似度")
                
                sim_color = "green" if result.similarity > 0.7 else ("orange" if result.similarity > 0.4 else "red")
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
                    <h2 style="color: {sim_color}; margin: 0;">{result.similarity:.2%}</h2>
                    <p style="margin: 5px 0 0 0;">文档相似度</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # 关键词对比
                st.markdown("### 🔑 关键词对比")
                
                kw_col1, kw_col2, kw_col3 = st.columns(3)
                
                with kw_col1:
                    st.markdown(f"**{result.doc1_name} 独特关键词**")
                    if result.doc1_unique_keywords:
                        for kw in result.doc1_unique_keywords[:10]:
                            st.markdown(f"- {kw}")
                    else:
                        st.info("无独特关键词")
                
                with kw_col2:
                    st.markdown("**共同关键词**")
                    if result.common_keywords:
                        for kw in result.common_keywords[:10]:
                            st.markdown(f"- {kw}")
                    else:
                        st.info("无共同关键词")
                
                with kw_col3:
                    st.markdown(f"**{result.doc2_name} 独特关键词**")
                    if result.doc2_unique_keywords:
                        for kw in result.doc2_unique_keywords[:10]:
                            st.markdown(f"- {kw}")
                    else:
                        st.info("无独特关键词")
    
    # ========== 关键词分析 ==========
    with tabs[2]:
        st.subheader("多文档关键词分析")
        
        # 选择多个文档
        selected_docs = st.multiselect(
            "选择要分析的文档（2-5个）",
            file_names,
            default=file_names[:min(3, len(file_names))],
            key="keyword_analysis_docs"
        )
        
        if len(selected_docs) < 2:
            st.warning("请至少选择2个文档")
        elif len(selected_docs) > 5:
            st.warning("建议选择不超过5个文档以获得更清晰的分析结果")
        else:
            top_n = st.slider("显示关键词数量", 5, 30, 15, key="kw_top_n")
            
            if st.button("🔑 分析关键词", type="primary", key="analyze_keywords"):
                doc_indices = [file_names.index(name) for name in selected_docs]
                
                with st.spinner("正在分析关键词..."):
                    # 共同关键词
                    common_kw = analyzer.find_common_keywords(doc_indices, top_n)
                    
                    # 各文档独特关键词
                    unique_kw = {}
                    for i, name in enumerate(selected_docs):
                        idx = file_names.index(name)
                        other_indices = [file_names.index(n) for n in selected_docs if n != name]
                        unique_kw[name] = analyzer.find_unique_keywords(idx, other_indices, top_n)
                    
                    st.session_state["keyword_analysis"] = {
                        "common": common_kw,
                        "unique": unique_kw,
                        "docs": selected_docs
                    }
                    log_message(f"分析了 {len(selected_docs)} 个文档的关键词")
            
            # 显示结果
            if st.session_state.get("keyword_analysis"):
                analysis = st.session_state["keyword_analysis"]
                
                # 共同关键词
                st.markdown("### 📌 共同关键词")
                if analysis["common"]:
                    # 显示为标签云样式
                    kw_html = " ".join([
                        f'<span style="background-color: #e1f5fe; padding: 5px 10px; margin: 3px; border-radius: 15px; display: inline-block;">{kw}</span>'
                        for kw in analysis["common"]
                    ])
                    st.markdown(kw_html, unsafe_allow_html=True)
                else:
                    st.info("这些文档没有共同关键词")
                
                st.markdown("---")
                
                # 各文档独特关键词
                st.markdown("### 🎯 各文档独特关键词")
                
                cols = st.columns(len(analysis["docs"]))
                for i, doc_name in enumerate(analysis["docs"]):
                    with cols[i]:
                        st.markdown(f"**{doc_name[:20]}...**" if len(doc_name) > 20 else f"**{doc_name}**")
                        unique = analysis["unique"].get(doc_name, [])
                        if unique:
                            for kw in unique[:10]:
                                st.markdown(f"- {kw}")
                        else:
                            st.info("无独特关键词")
                
                # 尝试绘制韦恩图（如果是2-3个文档）
                if 2 <= len(analysis["docs"]) <= 3:
                    st.markdown("---")
                    st.markdown("### 📊 关键词重叠韦恩图")
                    
                    try:
                        from matplotlib_venn import venn2, venn3
                        import matplotlib.pyplot as plt
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # 获取每个文档的词集合
                        doc_indices = [file_names.index(name) for name in analysis["docs"]]
                        word_sets = [set(texts[idx]) for idx in doc_indices]
                        
                        if len(analysis["docs"]) == 2:
                            venn2(word_sets, set_labels=analysis["docs"], ax=ax)
                        else:
                            venn3(word_sets, set_labels=analysis["docs"], ax=ax)
                        
                        ax.set_title("文档关键词重叠情况")
                        st.pyplot(fig)
                        plt.close()
                        
                    except ImportError:
                        st.info("💡 安装 matplotlib-venn 库可显示韦恩图: pip install matplotlib-venn")
    
    # ========== 相似段落 ==========
    with tabs[3]:
        st.subheader("相似段落检测")
        
        if not raw_texts:
            st.warning("⚠️ 需要原始文本数据才能进行段落比较")
            return
        
        # 选择文档
        col1, col2 = st.columns(2)
        
        with col1:
            seg_doc1 = st.selectbox(
                "选择第一个文档",
                file_names,
                index=0,
                key="seg_doc1"
            )
        
        with col2:
            default_idx = 1 if len(file_names) > 1 else 0
            seg_doc2 = st.selectbox(
                "选择第二个文档",
                file_names,
                index=default_idx,
                key="seg_doc2"
            )
        
        # 参数设置
        col3, col4 = st.columns(2)
        with col3:
            threshold = st.slider(
                "相似度阈值",
                0.3, 0.9, 0.5, 0.05,
                help="只显示相似度高于此阈值的段落对",
                key="seg_threshold"
            )
        with col4:
            segment_size = st.slider(
                "段落大小（字符数）",
                30, 200, 80, 10,
                help="用于分割文本的段落大小",
                key="seg_size"
            )
        
        if seg_doc1 == seg_doc2:
            st.warning("请选择两个不同的文档")
        else:
            if st.button("🔍 检测相似段落", type="primary", key="detect_segments"):
                doc1_idx = file_names.index(seg_doc1)
                doc2_idx = file_names.index(seg_doc2)
                
                with st.spinner("正在检测相似段落..."):
                    similar_segments = analyzer.find_similar_segments(
                        doc1_idx, doc2_idx, 
                        threshold=threshold,
                        segment_size=segment_size
                    )
                    st.session_state["similar_segments"] = {
                        "segments": similar_segments,
                        "doc1": seg_doc1,
                        "doc2": seg_doc2
                    }
                    log_message(f"检测到 {len(similar_segments)} 对相似段落")
            
            # 显示结果
            if st.session_state.get("similar_segments"):
                seg_data = st.session_state["similar_segments"]
                segments = seg_data["segments"]
                
                if segments:
                    st.success(f"找到 {len(segments)} 对相似段落")
                    
                    for i, (seg1, seg2, sim) in enumerate(segments):
                        with st.expander(f"相似段落 {i+1} (相似度: {sim:.2%})", expanded=(i < 3)):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**{seg_data['doc1']}**")
                                st.markdown(f'<div style="background-color: #fff3e0; padding: 10px; border-radius: 5px;">{seg1}</div>', 
                                          unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"**{seg_data['doc2']}**")
                                st.markdown(f'<div style="background-color: #e3f2fd; padding: 10px; border-radius: 5px;">{seg2}</div>', 
                                          unsafe_allow_html=True)
                else:
                    st.info("未找到相似度高于阈值的段落对，可以尝试降低相似度阈值")
    
    # ========== 导出 ==========
    with tabs[4]:
        st.subheader("导出比较分析结果")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            st.markdown("**📊 相似度矩阵**")
            if st.button("生成相似度矩阵", key="gen_sim_matrix"):
                with st.spinner("正在生成..."):
                    analyzer.calculate_similarity_matrix()
                    csv_content = analyzer.export_similarity_matrix()
                    if csv_content:
                        st.session_state["sim_matrix_csv"] = csv_content
                        st.success("相似度矩阵已生成")
            
            if st.session_state.get("sim_matrix_csv"):
                st.download_button(
                    label="📥 下载相似度矩阵CSV",
                    data=st.session_state["sim_matrix_csv"],
                    file_name="similarity_matrix.csv",
                    mime="text/csv",
                    key="download_sim_matrix"
                )
        
        with export_col2:
            st.markdown("**📋 比较分析报告**")
            if st.button("生成比较报告", key="gen_comparison_report"):
                with st.spinner("正在生成..."):
                    csv_content = analyzer.export_comparison()
                    if csv_content:
                        st.session_state["comparison_csv"] = csv_content
                        st.success("比较报告已生成")
            
            if st.session_state.get("comparison_csv"):
                st.download_button(
                    label="📥 下载比较报告CSV",
                    data=st.session_state["comparison_csv"],
                    file_name="comparison_report.csv",
                    mime="text/csv",
                    key="download_comparison"
                )
