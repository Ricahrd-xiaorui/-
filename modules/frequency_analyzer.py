# -*- coding: utf-8 -*-
"""
词频与共现分析模块 (Word Frequency and Co-occurrence Analysis Module)

本模块提供词频统计和词语共现分析功能，包括：
- 词频统计与排序
- 词性筛选
- 共现关系计算
- 共现网络数据转换
- 结果导出

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import pandas as pd

# 导入字体配置
try:
    from utils.font_config import get_plotly_font, get_label
except ImportError:
    def get_plotly_font():
        return "Arial, sans-serif"
    def get_label(cn, en):
        return cn


class FrequencyAnalyzer:
    """
    词频分析器 - 统计词语频率并支持词性筛选
    
    Attributes:
        texts: 分词后的文本列表（每个文本是词语列表）
        pos_tags: 词性标注列表（与texts对应，可选）
    
    Requirements: 2.1, 2.2
    """
    
    def __init__(self, texts: List[List[str]], pos_tags: Optional[List[List[str]]] = None):
        """
        初始化词频分析器
        
        Args:
            texts: 分词后的文本列表
            pos_tags: 词性标注列表（可选）
        """
        self.texts = texts if texts else []
        self.pos_tags = pos_tags
        self._word_frequency: Optional[Dict[str, int]] = None
        self._word_pos_map: Optional[Dict[str, Set[str]]] = None
    
    def _build_word_pos_map(self) -> Dict[str, Set[str]]:
        """
        构建词语到词性的映射
        
        Returns:
            Dict[str, Set[str]]: 词语 -> 词性集合
        """
        if self._word_pos_map is not None:
            return self._word_pos_map
        
        self._word_pos_map = defaultdict(set)
        
        if self.pos_tags:
            for text_idx, text in enumerate(self.texts):
                if text_idx < len(self.pos_tags):
                    pos_list = self.pos_tags[text_idx]
                    for word_idx, word in enumerate(text):
                        if word_idx < len(pos_list):
                            self._word_pos_map[word].add(pos_list[word_idx])
        
        return self._word_pos_map
    
    def calculate_word_frequency(self) -> Dict[str, int]:
        """
        计算所有词语的出现频率
        
        Returns:
            Dict[str, int]: 词语频率字典
        
        Requirements: 2.1
        """
        if self._word_frequency is not None:
            return self._word_frequency
        
        counter = Counter()
        for text in self.texts:
            counter.update(text)
        
        self._word_frequency = dict(counter)
        return self._word_frequency
    
    def get_total_word_count(self) -> int:
        """
        获取文本中的总词数
        
        Returns:
            int: 总词数
        """
        return sum(len(text) for text in self.texts)
    
    def filter_by_pos(self, pos_list: List[str]) -> Dict[str, int]:
        """
        按词性筛选词频统计结果
        
        Args:
            pos_list: 要保留的词性列表
        
        Returns:
            Dict[str, int]: 筛选后的词频字典
        
        Requirements: 2.2
        """
        if not self.pos_tags:
            # 没有词性标注时返回空字典
            return {}
        
        word_pos_map = self._build_word_pos_map()
        word_freq = self.calculate_word_frequency()
        
        # 筛选指定词性的词语
        pos_set = set(pos_list)
        filtered = {}
        
        for word, freq in word_freq.items():
            word_pos = word_pos_map.get(word, set())
            # 如果词语的任一词性在指定列表中，则保留
            if word_pos & pos_set:
                filtered[word] = freq
        
        return filtered
    
    def get_top_words(self, n: int, pos_filter: Optional[List[str]] = None) -> List[Tuple[str, int]]:
        """
        获取频率最高的n个词语
        
        Args:
            n: 返回的词语数量
            pos_filter: 词性筛选列表（可选）
        
        Returns:
            List[Tuple[str, int]]: (词语, 频率) 列表，按频率降序排列
        """
        if pos_filter:
            word_freq = self.filter_by_pos(pos_filter)
        else:
            word_freq = self.calculate_word_frequency()
        
        # 按频率降序排序
        sorted_words = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))
        return sorted_words[:n]
    
    def get_word_pos(self, word: str) -> Set[str]:
        """
        获取词语的词性标注
        
        Args:
            word: 词语
        
        Returns:
            Set[str]: 词性集合
        """
        word_pos_map = self._build_word_pos_map()
        return word_pos_map.get(word, set())
    
    def export_frequency_csv(self, include_pos: bool = False) -> str:
        """
        导出词频统计结果为CSV格式
        
        Args:
            include_pos: 是否包含词性信息
        
        Returns:
            str: CSV格式字符串
        
        Requirements: 2.7
        """
        word_freq = self.calculate_word_frequency()
        sorted_words = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))
        
        if include_pos and self.pos_tags:
            word_pos_map = self._build_word_pos_map()
            data = [
                {
                    "词语": word,
                    "频率": freq,
                    "词性": ",".join(sorted(word_pos_map.get(word, set())))
                }
                for word, freq in sorted_words
            ]
        else:
            data = [{"词语": word, "频率": freq} for word, freq in sorted_words]
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='utf-8-sig')


class CooccurrenceAnalyzer:
    """
    共现分析器 - 计算词语间的共现关系
    
    Attributes:
        texts: 分词后的文本列表
        window_size: 共现窗口大小
    
    Requirements: 2.3, 2.5
    """
    
    def __init__(self, texts: List[List[str]], window_size: int = 5):
        """
        初始化共现分析器
        
        Args:
            texts: 分词后的文本列表
            window_size: 共现窗口大小（默认5）
        """
        self.texts = texts if texts else []
        self.window_size = max(1, window_size)
        self._cooccurrence_matrix: Optional[Dict[Tuple[str, str], int]] = None
    
    def calculate_cooccurrence(self) -> Dict[Tuple[str, str], int]:
        """
        计算词语间的共现频率
        
        使用滑动窗口方法计算共现关系。
        共现对按字典序存储，确保 (A, B) 和 (B, A) 被视为同一对。
        
        Returns:
            Dict[Tuple[str, str], int]: 共现频率字典，键为词语对元组
        
        Requirements: 2.3
        """
        if self._cooccurrence_matrix is not None:
            return self._cooccurrence_matrix
        
        cooccurrence = Counter()
        
        for text in self.texts:
            text_len = len(text)
            for i, word1 in enumerate(text):
                # 在窗口范围内查找共现词
                window_end = min(i + self.window_size + 1, text_len)
                for j in range(i + 1, window_end):
                    word2 = text[j]
                    if word1 != word2:
                        # 按字典序排列，确保一致性
                        pair = tuple(sorted([word1, word2]))
                        cooccurrence[pair] += 1
        
        self._cooccurrence_matrix = dict(cooccurrence)
        return self._cooccurrence_matrix
    
    def filter_by_threshold(self, min_freq: int) -> Dict[Tuple[str, str], int]:
        """
        按最小频率阈值过滤共现结果
        
        Args:
            min_freq: 最小共现频率阈值
        
        Returns:
            Dict[Tuple[str, str], int]: 过滤后的共现频率字典
        
        Requirements: 2.5
        """
        cooccurrence = self.calculate_cooccurrence()
        return {
            pair: freq 
            for pair, freq in cooccurrence.items() 
            if freq >= min_freq
        }
    
    def get_top_cooccurrences(self, n: int, min_freq: int = 1) -> List[Tuple[Tuple[str, str], int]]:
        """
        获取共现频率最高的n对词语
        
        Args:
            n: 返回的词语对数量
            min_freq: 最小共现频率阈值
        
        Returns:
            List[Tuple[Tuple[str, str], int]]: ((词语1, 词语2), 频率) 列表
        """
        filtered = self.filter_by_threshold(min_freq)
        sorted_pairs = sorted(filtered.items(), key=lambda x: (-x[1], x[0]))
        return sorted_pairs[:n]
    
    def get_word_cooccurrences(self, word: str, min_freq: int = 1) -> Dict[str, int]:
        """
        获取指定词语的所有共现词及频率
        
        Args:
            word: 目标词语
            min_freq: 最小共现频率阈值
        
        Returns:
            Dict[str, int]: 共现词 -> 频率
        """
        cooccurrence = self.calculate_cooccurrence()
        result = {}
        
        for pair, freq in cooccurrence.items():
            if freq >= min_freq:
                if pair[0] == word:
                    result[pair[1]] = freq
                elif pair[1] == word:
                    result[pair[0]] = freq
        
        return result
    
    def to_network_data(self, min_freq: int = 1, max_nodes: int = 100) -> Tuple[List[dict], List[dict]]:
        """
        将共现数据转换为网络图数据格式
        
        Args:
            min_freq: 最小共现频率阈值
            max_nodes: 最大节点数量
        
        Returns:
            Tuple[List[dict], List[dict]]: (节点列表, 边列表)
            - 节点格式: {"id": str, "label": str, "size": int}
            - 边格式: {"source": str, "target": str, "weight": int}
        
        Requirements: 2.4
        """
        filtered = self.filter_by_threshold(min_freq)
        
        if not filtered:
            return [], []
        
        # 统计节点出现次数（用于确定节点大小）
        node_counts = Counter()
        for (word1, word2), freq in filtered.items():
            node_counts[word1] += freq
            node_counts[word2] += freq
        
        # 限制节点数量，保留出现频率最高的节点
        top_nodes = [word for word, _ in node_counts.most_common(max_nodes)]
        top_nodes_set = set(top_nodes)
        
        # 构建节点列表
        nodes = [
            {
                "id": word,
                "label": word,
                "size": node_counts[word]
            }
            for word in top_nodes
        ]
        
        # 构建边列表（只保留两端都在top_nodes中的边）
        edges = []
        for (word1, word2), freq in filtered.items():
            if word1 in top_nodes_set and word2 in top_nodes_set:
                edges.append({
                    "source": word1,
                    "target": word2,
                    "weight": freq
                })
        
        return nodes, edges
    
    def export_matrix_csv(self, min_freq: int = 1) -> str:
        """
        导出共现矩阵为CSV格式
        
        Args:
            min_freq: 最小共现频率阈值
        
        Returns:
            str: CSV格式字符串
        
        Requirements: 2.7
        """
        filtered = self.filter_by_threshold(min_freq)
        sorted_pairs = sorted(filtered.items(), key=lambda x: (-x[1], x[0]))
        
        data = [
            {"词语1": pair[0], "词语2": pair[1], "共现频率": freq}
            for pair, freq in sorted_pairs
        ]
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='utf-8-sig')
    
    def export_adjacency_matrix_csv(self, min_freq: int = 1, max_words: int = 50) -> str:
        """
        导出邻接矩阵格式的共现数据
        
        Args:
            min_freq: 最小共现频率阈值
            max_words: 最大词语数量
        
        Returns:
            str: CSV格式的邻接矩阵
        """
        filtered = self.filter_by_threshold(min_freq)
        
        if not filtered:
            return ""
        
        # 获取所有词语并限制数量
        all_words = set()
        for word1, word2 in filtered.keys():
            all_words.add(word1)
            all_words.add(word2)
        
        # 按总共现频率排序，取前max_words个
        word_freq = Counter()
        for (word1, word2), freq in filtered.items():
            word_freq[word1] += freq
            word_freq[word2] += freq
        
        top_words = [w for w, _ in word_freq.most_common(max_words)]
        top_words_set = set(top_words)
        
        # 构建邻接矩阵
        matrix = {word: {w: 0 for w in top_words} for word in top_words}
        
        for (word1, word2), freq in filtered.items():
            if word1 in top_words_set and word2 in top_words_set:
                matrix[word1][word2] = freq
                matrix[word2][word1] = freq
        
        # 转换为DataFrame
        df = pd.DataFrame(matrix, index=top_words, columns=top_words)
        return df.to_csv(encoding='utf-8-sig')




# ============================================================================
# Streamlit UI 渲染函数
# ============================================================================

def render_frequency_analyzer():
    """
    渲染词频分析模块UI
    
    Requirements: 2.4, 2.7
    """
    import streamlit as st
    from utils.session_state import log_message
    
    st.header("词频分析")
    
    # 功能介绍与操作手册
    with st.expander("📖 功能介绍与操作手册", expanded=False):
        st.markdown("""
        ## 📊 词频分析模块
        
        **功能概述**：统计文本中词语的出现频率，支持词性筛选和可视化展示。
        
        ---
        
        ### 🎯 使用场景
        
        | 场景 | 操作建议 | 应用 |
        |------|----------|------|
        | 了解文本主题 | 查看高频词 | 快速把握文本核心内容 |
        | 提取关键词 | 筛选名词 | 提取文本中的关键概念 |
        | 分析动作倾向 | 筛选动词 | 了解政策的行动导向 |
        | 情感分析准备 | 筛选形容词 | 为情感分析提供基础 |
        
        ---
        
        ### 📋 操作步骤
        
        **基础词频分析**：
        1. 设置显示词语数量（10-200）
        2. 选择词性筛选（可选）
        3. 点击"开始分析"
        4. 查看词频表格和图表
        5. 下载CSV文件
        
        **词性筛选分析**：
        1. 在词性筛选下拉框选择词性
        2. 可选：名词、动词、形容词、副词、自定义
        3. 自定义时输入词性标签（如n,v,a）
        4. 点击"开始分析"
        
        ---
        
        ### ⚙️ 参数说明
        
        | 参数 | 范围 | 默认值 | 说明 |
        |------|------|--------|------|
        | 显示词语数量 | 10-200 | 50 | 显示频率最高的N个词 |
        | 词性筛选 | 多选 | 全部 | 只统计指定词性的词语 |
        
        ---
        
        ### 🏷️ 词性标签说明
        
        | 词性 | 标签 | 示例 |
        |------|------|------|
        | 名词 | n, nr, ns, nt, nz | 政策、发展、创新 |
        | 动词 | v, vd, vn | 推进、实施、加强 |
        | 形容词 | a, ad, an | 重要、全面、深入 |
        | 副词 | d | 进一步、切实、全面 |
        
        ---
        
        ### 💡 使用建议
        
        **学术研究建议**：
        - 导出词频表作为论文附录
        - 结合词性筛选分析不同类型词语的分布
        - 高频词可作为主题分析的参考
        
        **政策分析建议**：
        - 名词高频词反映政策关注的领域
        - 动词高频词反映政策的行动导向
        - 形容词高频词反映政策的价值取向
        
        ---
        
        ### ❓ 常见问题
        
        **Q: 词性筛选不可用怎么办？**
        A: 词性筛选需要在预处理时启用词性标注功能。
        
        **Q: 如何判断词频分析结果的质量？**
        A: 高频词应该是有意义的实词，如果出现大量虚词，需要调整停用词设置。
        """)
    
    # 检查数据
    if not st.session_state.get("texts"):
        st.warning("请先在「文本预处理」标签页中完成文本预处理")
        return
    
    texts = st.session_state["texts"]
    pos_tags = st.session_state.get("pos_tags")
    
    # 创建分析器
    analyzer = FrequencyAnalyzer(texts, pos_tags)
    
    # 参数设置
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("显示词语数量", min_value=10, max_value=200, value=50, step=10)
    with col2:
        # 词性筛选（如果有词性标注）
        pos_filter = None
        if pos_tags:
            pos_options = ["全部", "名词(n)", "动词(v)", "形容词(a)", "副词(d)", "自定义"]
            selected_pos = st.selectbox("词性筛选", pos_options)
            
            if selected_pos == "名词(n)":
                pos_filter = ["n", "nr", "ns", "nt", "nz", "ng"]
            elif selected_pos == "动词(v)":
                pos_filter = ["v", "vd", "vn", "vg"]
            elif selected_pos == "形容词(a)":
                pos_filter = ["a", "ad", "an", "ag"]
            elif selected_pos == "副词(d)":
                pos_filter = ["d"]
            elif selected_pos == "自定义":
                custom_pos = st.text_input("输入词性标签（逗号分隔）", "n,v,a")
                pos_filter = [p.strip() for p in custom_pos.split(",") if p.strip()]
    
    # 执行分析
    if st.button("开始分析", key="freq_analyze_btn", type="primary"):
        with st.spinner("正在统计词频..."):
            # 获取词频
            top_words = analyzer.get_top_words(top_n, pos_filter)
            
            if top_words:
                # 保存到会话状态
                st.session_state["word_frequency"] = dict(top_words)
                
                # 显示统计信息
                total_words = analyzer.get_total_word_count()
                unique_words = len(analyzer.calculate_word_frequency())
                
                col1, col2, col3 = st.columns(3)
                col1.metric("总词数", f"{total_words:,}")
                col2.metric("不同词汇数", f"{unique_words:,}")
                col3.metric("显示词汇数", len(top_words))
                
                log_message(f"词频分析完成，共 {unique_words} 个不同词汇")
            else:
                st.warning("未找到符合条件的词语")
    
    # 显示结果
    if st.session_state.get("word_frequency"):
        word_freq = st.session_state["word_frequency"]
        
        # 创建标签页
        result_tabs = st.tabs(["📊 词频表格", "📈 词频图表", "💾 导出"])
        
        # 词频表格
        with result_tabs[0]:
            df = pd.DataFrame(
                list(word_freq.items()),
                columns=["词语", "频率"]
            )
            df["排名"] = range(1, len(df) + 1)
            df = df[["排名", "词语", "频率"]]
            
            st.dataframe(df, use_container_width=True, hide_index=True, height=400)
        
        # 词频图表
        with result_tabs[1]:
            try:
                import plotly.express as px
                
                # 取前30个词显示
                display_words = list(word_freq.items())[:30]
                chart_df = pd.DataFrame(display_words, columns=["词语", "频率"])
                
                fig = px.bar(
                    chart_df,
                    x="词语",
                    y="频率",
                    title="词频分布（前30个词）",
                    color="频率",
                    color_continuous_scale="Blues"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
            except ImportError:
                st.warning("需要安装plotly库才能显示图表: pip install plotly")
                # 使用Streamlit原生图表
                chart_df = pd.DataFrame(
                    list(word_freq.items())[:20],
                    columns=["词语", "频率"]
                )
                st.bar_chart(chart_df.set_index("词语"))
        
        # 导出
        with result_tabs[2]:
            csv_content = analyzer.export_frequency_csv(include_pos=bool(pos_tags))
            st.download_button(
                label="📥 下载词频表CSV",
                data=csv_content,
                file_name="word_frequency.csv",
                mime="text/csv"
            )


def render_cooccurrence_analyzer():
    """
    渲染词语共现分析模块UI
    
    Requirements: 2.4, 2.6, 2.7
    """
    import streamlit as st
    from utils.session_state import log_message
    
    st.header("词语共现分析")
    
    # 功能介绍与操作手册
    with st.expander("📖 功能介绍与操作手册", expanded=False):
        st.markdown("""
        ## 🔗 词语共现分析模块
        
        **功能概述**：分析词语间的共现关系，揭示词语之间的语义关联。
        
        ---
        
        ### 🎯 使用场景
        
        | 场景 | 关注点 | 应用 |
        |------|--------|------|
        | 概念关联分析 | 高频共现词对 | 发现概念之间的关联 |
        | 语义网络构建 | 共现网络图 | 可视化词语关系网络 |
        | 关键词扩展 | 特定词的共现词 | 扩展关键词列表 |
        | 主题发现 | 社区检测 | 发现潜在的主题聚类 |
        
        ---
        
        ### 📋 算法原理
        
        **滑动窗口共现计算**：
        - 在文本中设置固定大小的窗口
        - 统计窗口内词语对的共同出现次数
        - 窗口越大，捕获的关联越远
        
        **示例**：窗口大小=3，文本="政策 推动 创新 发展"
        - (政策, 推动)、(政策, 创新)、(推动, 创新)、(推动, 发展)、(创新, 发展)
        
        ---
        
        ### ⚙️ 参数说明
        
        | 参数 | 范围 | 默认值 | 说明 |
        |------|------|--------|------|
        | 共现窗口大小 | 2-20 | 5 | 计算共现的词语范围 |
        | 最小共现频率 | 1-50 | 2 | 过滤低频共现对 |
        | 最大节点数 | 20-200 | 50 | 网络图显示的最大节点数 |
        
        **参数调优建议**：
        - 窗口大小：句子级分析用3-5，段落级分析用10-15
        - 最小频率：数据量大时提高，数据量小时降低
        - 最大节点数：根据可视化清晰度调整
        
        ---
        
        ### 📋 操作步骤
        
        **基础共现分析**：
        1. 设置共现窗口大小
        2. 设置最小共现频率
        3. 设置最大节点数
        4. 点击"计算共现关系"
        5. 查看共现网络、表格、查询结果
        
        **网络图样式设置**：
        1. 展开"图表样式设置"
        2. 选择布局算法（推荐spring力导向）
        3. 调整节点和边的样式
        4. 可启用社区检测进行聚类着色
        
        **词语查询**：
        1. 切换到"词语查询"标签页
        2. 输入要查询的词语
        3. 查看该词的所有共现词
        
        ---
        
        ### 🎨 布局算法说明
        
        | 算法 | 特点 | 适用场景 |
        |------|------|----------|
        | spring (力导向) | 节点间模拟弹簧力 | 通用，推荐使用 |
        | kamada_kawai | 基于图论距离优化 | 小型网络 |
        | circular | 节点均匀分布在圆周 | 展示连接关系 |
        | shell | 按度数分层同心圆 | 展示层次结构 |
        
        ---
        
        ### 💡 使用建议
        
        **学术研究建议**：
        - 报告共现窗口大小和最小频率阈值
        - 导出共现矩阵用于后续网络分析
        - 使用社区检测发现词语聚类
        
        **可视化建议**：
        - 学术论文配图建议使用"学术蓝"或"黑白打印"配色
        - 调整节点大小和标签字体确保清晰可读
        - 可导出为高分辨率图片
        
        ---
        
        ### ❓ 常见问题
        
        **Q: 共现网络太密集怎么办？**
        A: 提高最小共现频率或减少最大节点数。
        
        **Q: 如何解读共现网络？**
        A: 连接越多的节点是核心词，连接越粗的边表示共现越频繁。
        
        **Q: 社区检测有什么用？**
        A: 可以发现语义相近的词语聚类，不同颜色代表不同社区。
        """)
    
    # 检查数据
    if not st.session_state.get("texts"):
        st.warning("请先在「文本预处理」标签页中完成文本预处理")
        return
    
    texts = st.session_state["texts"]
    
    # 参数设置
    st.subheader("参数设置")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        window_size = st.slider(
            "共现窗口大小",
            min_value=2,
            max_value=20,
            value=st.session_state.get("cooccurrence_window_size", 5),
            help="在多少个词的范围内计算共现关系"
        )
        st.session_state["cooccurrence_window_size"] = window_size
    
    with col2:
        min_freq = st.slider(
            "最小共现频率",
            min_value=1,
            max_value=50,
            value=st.session_state.get("cooccurrence_min_freq", 2),
            help="过滤共现频率低于此值的词语对"
        )
        st.session_state["cooccurrence_min_freq"] = min_freq
    
    with col3:
        max_nodes = st.slider(
            "最大节点数",
            min_value=20,
            max_value=200,
            value=50,
            help="网络图中显示的最大节点数量"
        )
    
    # 执行分析
    if st.button("计算共现关系", key="cooc_analyze_btn", type="primary"):
        with st.spinner("正在计算共现关系..."):
            analyzer = CooccurrenceAnalyzer(texts, window_size)
            cooccurrence = analyzer.filter_by_threshold(min_freq)
            
            if cooccurrence:
                # 保存到会话状态
                st.session_state["cooccurrence_matrix"] = cooccurrence
                st.session_state["cooccurrence_analyzer"] = analyzer
                
                st.success(f"共现分析完成，找到 {len(cooccurrence)} 对共现词语")
                log_message(f"共现分析完成，窗口大小={window_size}，最小频率={min_freq}")
            else:
                st.warning("未找到符合条件的共现词语对，请尝试降低最小共现频率")
    
    # 显示结果
    if st.session_state.get("cooccurrence_matrix"):
        cooccurrence = st.session_state["cooccurrence_matrix"]
        analyzer = st.session_state.get("cooccurrence_analyzer")
        
        if analyzer is None:
            analyzer = CooccurrenceAnalyzer(texts, window_size)
        
        # 创建标签页
        result_tabs = st.tabs(["🕸️ 共现网络", "📊 共现表格", "🔍 词语查询", "💾 导出"])
        
        # 共现网络
        with result_tabs[0]:
            st.subheader("共现网络图")
            
            # 学术论文风格设置
            with st.expander("🎨 图表样式设置", expanded=True):
                
                # 第一行：布局和配色
                st.markdown("**📐 布局与配色**")
                layout_col1, layout_col2 = st.columns(2)
                
                with layout_col1:
                    layout_algorithm = st.selectbox(
                        "布局算法",
                        [
                            "spring (力导向)",
                            "kamada_kawai (路径优化)",
                            "fruchterman_reingold (FR算法)",
                            "circular (圆形)",
                            "shell (同心圆)",
                            "spectral (谱布局)",
                            "random (随机)",
                            "spiral (螺旋)"
                        ],
                        index=0,
                        help="""
• spring: 力导向布局，节点间模拟弹簧力（推荐）
• kamada_kawai: 基于图论距离优化，适合小型网络
• fruchterman_reingold: 经典FR力导向算法
• circular: 节点均匀分布在圆周上
• shell: 按度数分层的同心圆布局
• spectral: 基于图拉普拉斯矩阵的谱布局
• random: 随机布局，可用于对比
• spiral: 螺旋形布局，适合展示层次
                        """
                    )
                
                with layout_col2:
                    color_scheme = st.selectbox(
                        "配色方案",
                        ["学术蓝", "经典灰", "暖色调", "冷色调", "彩虹", "黑白打印"],
                        index=0,
                        help="选择适合学术论文的配色方案，黑白打印适合灰度印刷"
                    )
                
                st.markdown("---")
                
                # 第二行：节点设置
                st.markdown("**⭕ 节点设置**")
                node_col1, node_col2, node_col3, node_col4 = st.columns(4)
                
                with node_col1:
                    node_size_mode = st.selectbox(
                        "节点大小依据",
                        ["度数 (连接数)", "权重 (共现总频率)", "统一大小"],
                        index=0,
                        help="选择节点大小的计算方式"
                    )
                
                with node_col2:
                    node_size_scale = st.slider(
                        "节点大小比例",
                        min_value=0.3,
                        max_value=3.0,
                        value=1.0,
                        step=0.1
                    )
                
                with node_col3:
                    node_shape = st.selectbox(
                        "节点形状",
                        ["圆形", "方形", "菱形", "三角形"],
                        index=0
                    )
                
                with node_col4:
                    node_opacity = st.slider(
                        "节点透明度",
                        min_value=0.3,
                        max_value=1.0,
                        value=0.9,
                        step=0.1
                    )
                
                st.markdown("---")
                
                # 第三行：边设置
                st.markdown("**➖ 边线设置**")
                edge_col1, edge_col2, edge_col3, edge_col4 = st.columns(4)
                
                with edge_col1:
                    edge_width_mode = st.selectbox(
                        "边线粗细依据",
                        ["权重 (共现频率)", "统一粗细"],
                        index=0,
                        help="选择边线粗细的计算方式"
                    )
                
                with edge_col2:
                    edge_width_scale = st.slider(
                        "边线粗细比例",
                        min_value=0.3,
                        max_value=3.0,
                        value=1.0,
                        step=0.1
                    )
                
                with edge_col3:
                    edge_style = st.selectbox(
                        "边线样式",
                        ["实线", "虚线", "点线"],
                        index=0
                    )
                
                with edge_col4:
                    edge_opacity = st.slider(
                        "边线透明度",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.6,
                        step=0.1
                    )
                
                st.markdown("---")
                
                # 第四行：标签设置
                st.markdown("**🏷️ 标签设置**")
                label_col1, label_col2, label_col3, label_col4 = st.columns(4)
                
                with label_col1:
                    show_node_labels = st.checkbox(
                        "显示节点标签",
                        value=True
                    )
                
                with label_col2:
                    font_size = st.slider(
                        "标签字体大小",
                        min_value=6,
                        max_value=24,
                        value=11
                    )
                
                with label_col3:
                    label_position = st.selectbox(
                        "标签位置",
                        ["上方", "下方", "右侧", "左侧", "居中"],
                        index=0
                    )
                
                with label_col4:
                    show_edge_labels = st.checkbox(
                        "显示边权重",
                        value=False
                    )
                
                st.markdown("---")
                
                # 第五行：高级设置
                st.markdown("**⚙️ 高级设置**")
                adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
                
                with adv_col1:
                    show_colorbar = st.checkbox(
                        "显示颜色图例",
                        value=True,
                        help="显示节点颜色对应的数值图例"
                    )
                
                with adv_col2:
                    show_title = st.checkbox(
                        "显示图表标题",
                        value=True
                    )
                
                with adv_col3:
                    transparent_bg = st.checkbox(
                        "透明背景",
                        value=False,
                        help="导出时使用透明背景"
                    )
                
                with adv_col4:
                    show_stats_annotation = st.checkbox(
                        "显示统计信息",
                        value=True,
                        help="在图表底部显示节点数、边数等"
                    )
                
                # 第六行：社区检测
                st.markdown("---")
                st.markdown("**🔬 社区检测（聚类着色）**")
                community_col1, community_col2 = st.columns(2)
                
                with community_col1:
                    enable_community = st.checkbox(
                        "启用社区检测",
                        value=False,
                        help="使用Louvain算法检测网络社区，不同社区用不同颜色标注"
                    )
                
                with community_col2:
                    if enable_community:
                        community_resolution = st.slider(
                            "社区粒度",
                            min_value=0.5,
                            max_value=2.0,
                            value=1.0,
                            step=0.1,
                            help="值越大，检测到的社区越多越小"
                        )
                    else:
                        community_resolution = 1.0
                
                # 自定义标题
                if show_title:
                    custom_title = st.text_input(
                        "自定义图表标题",
                        value="词语共现网络图",
                        help="输入您想要显示的图表标题"
                    )
                else:
                    custom_title = ""
            
            nodes, edges = analyzer.to_network_data(min_freq, max_nodes)
            
            if nodes and edges:
                try:
                    import plotly.graph_objects as go
                    import networkx as nx
                    
                    # 构建NetworkX图
                    G = nx.Graph()
                    for node in nodes:
                        G.add_node(node["id"], size=node["size"])
                    for edge in edges:
                        G.add_edge(edge["source"], edge["target"], weight=edge["weight"])
                    
                    # 根据选择的布局算法计算位置
                    layout_name = layout_algorithm.split(" ")[0]  # 提取算法名称
                    
                    if layout_name == "spring":
                        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
                    elif layout_name == "kamada_kawai":
                        try:
                            pos = nx.kamada_kawai_layout(G)
                        except:
                            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
                            st.info("Kamada-Kawai布局计算失败，已切换到Spring布局")
                    elif layout_name == "fruchterman_reingold":
                        pos = nx.fruchterman_reingold_layout(G, k=2, iterations=100, seed=42)
                    elif layout_name == "circular":
                        pos = nx.circular_layout(G)
                    elif layout_name == "shell":
                        # 按度数分层
                        degrees = dict(G.degree())
                        if degrees:
                            max_deg = max(degrees.values())
                            shells = [[] for _ in range(min(5, max_deg + 1))]
                            for node, deg in degrees.items():
                                shell_idx = min(deg * 4 // (max_deg + 1), len(shells) - 1)
                                shells[shell_idx].append(node)
                            shells = [s for s in shells if s]  # 移除空层
                            if shells:
                                pos = nx.shell_layout(G, nlist=shells)
                            else:
                                pos = nx.shell_layout(G)
                        else:
                            pos = nx.shell_layout(G)
                    elif layout_name == "spectral":
                        try:
                            pos = nx.spectral_layout(G)
                        except:
                            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
                            st.info("谱布局计算失败，已切换到Spring布局")
                    elif layout_name == "random":
                        pos = nx.random_layout(G, seed=42)
                    elif layout_name == "spiral":
                        pos = nx.spiral_layout(G)
                    else:
                        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
                    
                    # 配色方案定义
                    color_schemes = {
                        "学术蓝": {
                            "node_colorscale": [[0, "#E3F2FD"], [0.5, "#2196F3"], [1, "#0D47A1"]],
                            "edge_color": "#90A4AE",
                            "bg_color": "white" if not transparent_bg else "rgba(0,0,0,0)",
                            "font_color": "#263238",
                            "community_colors": ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#C2185B", "#00796B", "#5D4037", "#455A64"]
                        },
                        "经典灰": {
                            "node_colorscale": [[0, "#ECEFF1"], [0.5, "#607D8B"], [1, "#263238"]],
                            "edge_color": "#B0BEC5",
                            "bg_color": "white" if not transparent_bg else "rgba(0,0,0,0)",
                            "font_color": "#37474F",
                            "community_colors": ["#37474F", "#546E7A", "#78909C", "#90A4AE", "#B0BEC5", "#CFD8DC", "#455A64", "#263238"]
                        },
                        "暖色调": {
                            "node_colorscale": [[0, "#FFF3E0"], [0.5, "#FF9800"], [1, "#E65100"]],
                            "edge_color": "#FFCC80",
                            "bg_color": "white" if not transparent_bg else "rgba(0,0,0,0)",
                            "font_color": "#BF360C",
                            "community_colors": ["#E65100", "#F57C00", "#FF9800", "#FFA726", "#FFB74D", "#FFCC80", "#D84315", "#BF360C"]
                        },
                        "冷色调": {
                            "node_colorscale": [[0, "#E8F5E9"], [0.5, "#4CAF50"], [1, "#1B5E20"]],
                            "edge_color": "#A5D6A7",
                            "bg_color": "white" if not transparent_bg else "rgba(0,0,0,0)",
                            "font_color": "#1B5E20",
                            "community_colors": ["#1B5E20", "#2E7D32", "#388E3C", "#43A047", "#4CAF50", "#66BB6A", "#81C784", "#A5D6A7"]
                        },
                        "彩虹": {
                            "node_colorscale": "Viridis",
                            "edge_color": "#9E9E9E",
                            "bg_color": "white" if not transparent_bg else "rgba(0,0,0,0)",
                            "font_color": "#424242",
                            "community_colors": ["#E91E63", "#9C27B0", "#673AB7", "#3F51B5", "#2196F3", "#00BCD4", "#4CAF50", "#FFEB3B"]
                        },
                        "黑白打印": {
                            "node_colorscale": [[0, "#E0E0E0"], [0.5, "#757575"], [1, "#212121"]],
                            "edge_color": "#9E9E9E",
                            "bg_color": "white" if not transparent_bg else "rgba(0,0,0,0)",
                            "font_color": "#212121",
                            "community_colors": ["#212121", "#424242", "#616161", "#757575", "#9E9E9E", "#BDBDBD", "#E0E0E0", "#F5F5F5"]
                        }
                    }
                    
                    scheme = color_schemes.get(color_scheme, color_schemes["学术蓝"])
                    
                    # 社区检测
                    community_labels = {}
                    num_communities = 0
                    if enable_community:
                        try:
                            from networkx.algorithms import community as nx_community
                            communities = nx_community.louvain_communities(G, resolution=community_resolution, seed=42)
                            for idx, comm in enumerate(communities):
                                for node in comm:
                                    community_labels[node] = idx
                            num_communities = len(communities)
                        except Exception as e:
                            st.warning(f"社区检测失败: {e}")
                            enable_community = False
                    
                    # 计算节点度数和权重
                    node_degrees = dict(G.degree())
                    max_degree = max(node_degrees.values()) if node_degrees else 1
                    
                    # 计算节点总权重
                    node_weights = {}
                    for node in G.nodes():
                        node_weights[node] = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
                    max_node_weight = max(node_weights.values()) if node_weights else 1
                    
                    # 计算边权重范围
                    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
                    max_weight = max(edge_weights) if edge_weights else 1
                    min_weight = min(edge_weights) if edge_weights else 1
                    
                    # 边线样式映射
                    edge_dash_map = {
                        "实线": "solid",
                        "虚线": "dash",
                        "点线": "dot"
                    }
                    edge_dash = edge_dash_map.get(edge_style, "solid")
                    
                    # 创建边的轨迹（根据权重调整粗细）
                    edge_traces = []
                    edge_annotations = []
                    
                    for edge in G.edges(data=True):
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        weight = edge[2].get('weight', 1)
                        
                        # 根据权重计算边的粗细
                        if edge_width_mode == "权重 (共现频率)":
                            if max_weight > min_weight:
                                normalized_weight = (weight - min_weight) / (max_weight - min_weight)
                            else:
                                normalized_weight = 0.5
                            edge_width = (0.5 + normalized_weight * 2.5) * edge_width_scale
                        else:
                            edge_width = 1.5 * edge_width_scale
                        
                        # 边颜色（考虑透明度）
                        edge_color_with_opacity = scheme["edge_color"]
                        if edge_opacity < 1.0:
                            # 转换为rgba
                            hex_color = scheme["edge_color"].lstrip('#')
                            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                            edge_color_with_opacity = f"rgba({r},{g},{b},{edge_opacity})"
                        
                        edge_trace = go.Scatter(
                            x=[x0, x1, None],
                            y=[y0, y1, None],
                            line=dict(width=edge_width, color=edge_color_with_opacity, dash=edge_dash),
                            hoverinfo='text',
                            hovertext=f"{edge[0]} - {edge[1]}: {weight}",
                            mode='lines',
                            showlegend=False
                        )
                        edge_traces.append(edge_trace)
                        
                        # 边标签（如果启用）
                        if show_edge_labels:
                            mid_x = (x0 + x1) / 2
                            mid_y = (y0 + y1) / 2
                            edge_annotations.append(
                                dict(
                                    x=mid_x,
                                    y=mid_y,
                                    text=str(weight),
                                    showarrow=False,
                                    font=dict(size=font_size - 2, color=scheme["font_color"]),
                                    bgcolor="rgba(255,255,255,0.7)"
                                )
                            )
                    
                    # 节点形状映射
                    node_symbol_map = {
                        "圆形": "circle",
                        "方形": "square",
                        "菱形": "diamond",
                        "三角形": "triangle-up"
                    }
                    node_symbol = node_symbol_map.get(node_shape, "circle")
                    
                    # 标签位置映射
                    label_pos_map = {
                        "上方": "top center",
                        "下方": "bottom center",
                        "右侧": "middle right",
                        "左侧": "middle left",
                        "居中": "middle center"
                    }
                    text_position = label_pos_map.get(label_position, "top center")
                    
                    # 创建节点的轨迹
                    node_x = []
                    node_y = []
                    node_text = []
                    node_hover = []
                    node_size = []
                    node_color = []
                    
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(node if show_node_labels else "")
                        
                        degree = node_degrees[node]
                        total_weight = node_weights[node]
                        
                        # 悬停信息
                        hover_text = f"{node}<br>连接数: {degree}<br>总共现频率: {total_weight}"
                        if enable_community and node in community_labels:
                            hover_text += f"<br>社区: {community_labels[node] + 1}"
                        node_hover.append(hover_text)
                        
                        # 节点大小
                        if node_size_mode == "度数 (连接数)":
                            base_size = 15 + (degree / max_degree) * 35
                        elif node_size_mode == "权重 (共现总频率)":
                            base_size = 15 + (total_weight / max_node_weight) * 35
                        else:  # 统一大小
                            base_size = 25
                        node_size.append(base_size * node_size_scale)
                        
                        # 节点颜色
                        if enable_community and node in community_labels:
                            node_color.append(community_labels[node])
                        else:
                            node_color.append(degree)
                    
                    # 确定节点颜色方案
                    if enable_community and num_communities > 0:
                        # 使用社区颜色
                        community_color_list = scheme["community_colors"]
                        node_colors_final = [community_color_list[c % len(community_color_list)] for c in node_color]
                        colorbar_config = None  # 社区模式不显示连续色条
                        use_discrete_colors = True
                    else:
                        node_colors_final = node_color
                        colorbar_config = dict(
                            thickness=15,
                            title=dict(
                                text='连接数' if node_size_mode == "度数 (连接数)" else '共现频率',
                                font=dict(size=12, family=get_plotly_font())
                            ),
                            xanchor='left',
                            tickfont=dict(size=10)
                        ) if show_colorbar else None
                        use_discrete_colors = False
                    
                    node_trace = go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode='markers+text' if show_node_labels else 'markers',
                        hoverinfo='text',
                        hovertext=node_hover,
                        text=node_text,
                        textposition=text_position,
                        textfont=dict(
                            size=font_size,
                            color=scheme["font_color"],
                            family=get_plotly_font()
                        ),
                        marker=dict(
                            showscale=show_colorbar and not use_discrete_colors,
                            colorscale=scheme["node_colorscale"] if not use_discrete_colors else None,
                            size=node_size,
                            color=node_colors_final,
                            colorbar=colorbar_config,
                            symbol=node_symbol,
                            opacity=node_opacity,
                            line=dict(width=1.5, color='white')
                        )
                    )
                    
                    # 创建图形
                    annotations_list = edge_annotations if show_edge_labels else []
                    
                    # 添加统计信息注释
                    if show_stats_annotation:
                        stats_text = f"节点数: {len(nodes)} | 边数: {len(edges)} | 布局: {layout_name}"
                        if enable_community and num_communities > 0:
                            stats_text += f" | 社区数: {num_communities}"
                        annotations_list.append(
                            dict(
                                text=stats_text,
                                xref="paper", yref="paper",
                                x=0, y=-0.02,
                                showarrow=False,
                                font=dict(size=10, color=scheme["font_color"]),
                                align="left"
                            )
                        )
                    
                    fig = go.Figure(
                        data=edge_traces + [node_trace],
                        layout=go.Layout(
                            title=dict(
                                text=custom_title if show_title else "",
                                font=dict(size=16, family=get_plotly_font(), color=scheme["font_color"]),
                                x=0.5,
                                xanchor='center'
                            ) if show_title else None,
                            showlegend=False,
                            hovermode='closest',
                            xaxis=dict(
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                                showline=False
                            ),
                            yaxis=dict(
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                                showline=False
                            ),
                            plot_bgcolor=scheme["bg_color"],
                            paper_bgcolor=scheme["bg_color"],
                            height=700,
                            margin=dict(l=20, r=20, t=50 if show_title else 20, b=40 if show_stats_annotation else 20),
                            annotations=annotations_list
                        )
                    )
                    
                    # 如果启用社区检测，添加社区图例
                    if enable_community and num_communities > 0:
                        for i in range(min(num_communities, 8)):
                            fig.add_trace(go.Scatter(
                                x=[None], y=[None],
                                mode='markers',
                                marker=dict(size=10, color=scheme["community_colors"][i]),
                                name=f'社区 {i+1}',
                                showlegend=True
                            ))
                        fig.update_layout(
                            showlegend=True,
                            legend=dict(
                                title="社区",
                                orientation="v",
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=1.02,
                                font=dict(size=10)
                            )
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示网络统计
                    st.markdown("---")
                    st.markdown("**📊 网络统计指标**")
                    
                    # 基础指标
                    avg_degree = sum(node_degrees.values()) / len(node_degrees) if node_degrees else 0
                    density = nx.density(G)
                    
                    stat_row1 = st.columns(4)
                    stat_row1[0].metric("节点数", len(nodes))
                    stat_row1[1].metric("边数", len(edges))
                    stat_row1[2].metric("平均度数", f"{avg_degree:.2f}")
                    stat_row1[3].metric("网络密度", f"{density:.4f}")
                    
                    # 高级指标（可展开）
                    with st.expander("🔬 高级网络指标", expanded=False):
                        try:
                            # 计算更多网络指标
                            adv_col1, adv_col2 = st.columns(2)
                            
                            with adv_col1:
                                st.markdown("**中心性指标**")
                                
                                # 度中心性
                                degree_centrality = nx.degree_centrality(G)
                                top_degree = sorted(degree_centrality.items(), key=lambda x: -x[1])[:5]
                                st.markdown("度中心性 Top 5:")
                                for word, cent in top_degree:
                                    st.markdown(f"- {word}: {cent:.4f}")
                                
                                # 介数中心性
                                if len(G.nodes()) <= 100:  # 大网络计算较慢
                                    betweenness = nx.betweenness_centrality(G)
                                    top_between = sorted(betweenness.items(), key=lambda x: -x[1])[:5]
                                    st.markdown("介数中心性 Top 5:")
                                    for word, cent in top_between:
                                        st.markdown(f"- {word}: {cent:.4f}")
                            
                            with adv_col2:
                                st.markdown("**网络结构指标**")
                                
                                # 聚类系数
                                avg_clustering = nx.average_clustering(G)
                                st.markdown(f"平均聚类系数: {avg_clustering:.4f}")
                                
                                # 连通分量
                                num_components = nx.number_connected_components(G)
                                st.markdown(f"连通分量数: {num_components}")
                                
                                # 最大连通分量大小
                                if num_components > 0:
                                    largest_cc = max(nx.connected_components(G), key=len)
                                    st.markdown(f"最大连通分量节点数: {len(largest_cc)}")
                                
                                # 社区数（如果启用）
                                if enable_community and num_communities > 0:
                                    st.markdown(f"检测到的社区数: {num_communities}")
                                    
                                    # 模块度
                                    try:
                                        from networkx.algorithms import community as nx_community
                                        communities_list = list(nx_community.louvain_communities(G, resolution=community_resolution, seed=42))
                                        modularity = nx_community.modularity(G, communities_list)
                                        st.markdown(f"模块度 (Modularity): {modularity:.4f}")
                                    except:
                                        pass
                        
                        except Exception as e:
                            st.warning(f"部分高级指标计算失败: {e}")
                    
                    # 导出高分辨率图片
                    st.markdown("---")
                    st.markdown("**📥 导出高分辨率图片**")
                    export_col1, export_col2, export_col3 = st.columns(3)
                    
                    with export_col1:
                        export_format = st.selectbox(
                            "导出格式",
                            ["PNG (推荐)", "SVG (矢量图)", "PDF"],
                            help="PNG适合一般用途，SVG适合需要缩放的场景，PDF适合直接插入论文"
                        )
                    
                    with export_col2:
                        export_dpi = st.selectbox(
                            "分辨率 (DPI)",
                            [150, 300, 600],
                            index=1,
                            help="300 DPI适合大多数学术期刊要求"
                        )
                    
                    with export_col3:
                        export_size = st.selectbox(
                            "图片尺寸",
                            ["标准 (1200×900)", "大 (1600×1200)", "超大 (2400×1800)"],
                            index=0,
                            help="选择导出图片的尺寸"
                        )
                    
                    if st.button("🖼️ 生成高分辨率图片", key="export_network_img"):
                        try:
                            import io
                            
                            # 解析尺寸
                            size_map = {
                                "标准 (1200×900)": (1200, 900),
                                "大 (1600×1200)": (1600, 1200),
                                "超大 (2400×1800)": (2400, 1800)
                            }
                            export_width, export_height = size_map.get(export_size, (1200, 900))
                            scale = export_dpi / 72  # 72是默认DPI
                            
                            if "PNG" in export_format:
                                img_bytes = fig.to_image(
                                    format="png",
                                    width=export_width,
                                    height=export_height,
                                    scale=scale
                                )
                                st.download_button(
                                    label="📥 下载 PNG 图片",
                                    data=img_bytes,
                                    file_name="cooccurrence_network.png",
                                    mime="image/png"
                                )
                            elif "SVG" in export_format:
                                img_bytes = fig.to_image(
                                    format="svg",
                                    width=export_width,
                                    height=export_height
                                )
                                st.download_button(
                                    label="📥 下载 SVG 图片",
                                    data=img_bytes,
                                    file_name="cooccurrence_network.svg",
                                    mime="image/svg+xml"
                                )
                            elif "PDF" in export_format:
                                img_bytes = fig.to_image(
                                    format="pdf",
                                    width=export_width,
                                    height=export_height
                                )
                                st.download_button(
                                    label="📥 下载 PDF 文件",
                                    data=img_bytes,
                                    file_name="cooccurrence_network.pdf",
                                    mime="application/pdf"
                                )
                            
                            st.success("图片生成成功！点击上方按钮下载")
                            
                        except Exception as e:
                            st.warning(f"图片导出需要安装 kaleido 库: pip install kaleido")
                            st.info("您也可以右键点击图表选择「另存为图片」")
                    
                except ImportError:
                    st.warning("需要安装plotly和networkx库才能显示网络图")
                    st.code("pip install plotly networkx")
            else:
                st.info("没有足够的数据生成网络图")
        
        # 共现表格
        with result_tabs[1]:
            st.subheader("共现词语对")
            
            # 排序并显示
            sorted_cooc = sorted(cooccurrence.items(), key=lambda x: -x[1])
            
            df = pd.DataFrame([
                {"词语1": pair[0], "词语2": pair[1], "共现频率": freq}
                for pair, freq in sorted_cooc
            ])
            
            st.dataframe(df, use_container_width=True, hide_index=True, height=400)
        
        # 词语查询
        with result_tabs[2]:
            st.subheader("查询词语共现")
            
            query_word = st.text_input("输入要查询的词语")
            
            if query_word:
                word_cooc = analyzer.get_word_cooccurrences(query_word, min_freq)
                
                if word_cooc:
                    st.success(f"找到 {len(word_cooc)} 个与「{query_word}」共现的词语")
                    
                    sorted_cooc = sorted(word_cooc.items(), key=lambda x: -x[1])
                    df = pd.DataFrame(sorted_cooc, columns=["共现词语", "共现频率"])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info(f"未找到与「{query_word}」共现的词语")
        
        # 导出
        with result_tabs[3]:
            st.subheader("导出共现数据")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**共现词语对列表**")
                csv_content = analyzer.export_matrix_csv(min_freq)
                st.download_button(
                    label="📥 下载共现列表CSV",
                    data=csv_content,
                    file_name="cooccurrence_list.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.markdown("**邻接矩阵格式**")
                adj_csv = analyzer.export_adjacency_matrix_csv(min_freq, max_nodes)
                if adj_csv:
                    st.download_button(
                        label="📥 下载邻接矩阵CSV",
                        data=adj_csv,
                        file_name="cooccurrence_matrix.csv",
                        mime="text/csv"
                    )
