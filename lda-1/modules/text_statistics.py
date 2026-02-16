# -*- coding: utf-8 -*-
"""
文本统计与可读性分析模块 (Text Statistics and Readability Analysis Module)

本模块提供文本的详细统计信息和可读性指标计算，包括：
- 字符数、词语数、句子数、段落数统计
- 平均句长、平均词长、词汇丰富度(TTR)等指标
- 可读性指数计算
- 多文档统计对比分析

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class TextStatistics:
    """
    文本统计分析器 - 计算单个文本的各项统计指标
    
    Attributes:
        raw_text: 原始文本
        tokenized_text: 分词后的词语列表
    """
    raw_text: str
    tokenized_text: List[str] = field(default_factory=list)
    
    # 缓存计算结果
    _char_count: Optional[int] = field(default=None, repr=False)
    _word_count: Optional[int] = field(default=None, repr=False)
    _sentence_count: Optional[int] = field(default=None, repr=False)
    _paragraph_count: Optional[int] = field(default=None, repr=False)
    _sentences: Optional[List[str]] = field(default=None, repr=False)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.raw_text is None:
            self.raw_text = ""
        if self.tokenized_text is None:
            self.tokenized_text = []
    
    def _split_sentences(self) -> List[str]:
        """
        将文本分割为句子
        
        Returns:
            List[str]: 句子列表
        """
        if self._sentences is not None:
            return self._sentences
        
        if not self.raw_text:
            self._sentences = []
            return self._sentences
        
        # 中文句子分隔符：句号、问号、感叹号、分号
        # 同时支持中英文标点
        pattern = r'[。！？；.!?;]+'
        
        # 分割句子
        sentences = re.split(pattern, self.raw_text)
        
        # 过滤空句子并去除首尾空白
        self._sentences = [s.strip() for s in sentences if s.strip()]
        
        return self._sentences
    
    def _split_paragraphs(self) -> List[str]:
        """
        将文本分割为段落
        
        Returns:
            List[str]: 段落列表
        """
        if not self.raw_text:
            return []
        
        # 按换行符分割段落
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', self.raw_text)
        
        # 如果没有双换行，尝试单换行
        if len(paragraphs) <= 1:
            paragraphs = self.raw_text.split('\n')
        
        # 过滤空段落
        return [p.strip() for p in paragraphs if p.strip()]
    
    def count_characters(self) -> int:
        """
        统计字符数（不含空白字符）
        
        Returns:
            int: 字符数
        """
        if self._char_count is not None:
            return self._char_count
        
        # 移除所有空白字符后计数
        text_no_space = re.sub(r'\s', '', self.raw_text)
        self._char_count = len(text_no_space)
        return self._char_count
    
    def count_words(self) -> int:
        """
        统计词语数
        
        Returns:
            int: 词语数
        """
        if self._word_count is not None:
            return self._word_count
        
        self._word_count = len(self.tokenized_text)
        return self._word_count
    
    def count_sentences(self) -> int:
        """
        统计句子数
        
        Returns:
            int: 句子数
        """
        if self._sentence_count is not None:
            return self._sentence_count
        
        sentences = self._split_sentences()
        self._sentence_count = len(sentences) if sentences else 1  # 至少1句
        return self._sentence_count
    
    def count_paragraphs(self) -> int:
        """
        统计段落数
        
        Returns:
            int: 段落数
        """
        if self._paragraph_count is not None:
            return self._paragraph_count
        
        paragraphs = self._split_paragraphs()
        self._paragraph_count = len(paragraphs) if paragraphs else 1  # 至少1段
        return self._paragraph_count
    
    def calculate_avg_sentence_length(self) -> float:
        """
        计算平均句长（每句词数）
        
        Returns:
            float: 平均句长
        """
        word_count = self.count_words()
        sentence_count = self.count_sentences()
        
        if sentence_count == 0:
            return 0.0
        
        return word_count / sentence_count
    
    def calculate_avg_word_length(self) -> float:
        """
        计算平均词长（每词字符数）
        
        Returns:
            float: 平均词长
        """
        if not self.tokenized_text:
            return 0.0
        
        total_chars = sum(len(word) for word in self.tokenized_text)
        return total_chars / len(self.tokenized_text)
    
    def calculate_ttr(self) -> float:
        """
        计算词汇丰富度 (Type-Token Ratio)
        TTR = 不同词汇数 / 总词汇数
        
        Returns:
            float: TTR值，范围(0, 1]
        """
        if not self.tokenized_text:
            return 0.0
        
        unique_words = set(self.tokenized_text)
        total_words = len(self.tokenized_text)
        
        if total_words == 0:
            return 0.0
        
        return len(unique_words) / total_words
    
    def calculate_readability_index(self) -> float:
        """
        计算可读性指数
        
        使用改进的中文可读性公式：
        RI = 0.5 * (平均句长 + 100 * (1 - TTR))
        
        值越低表示越易读
        
        Returns:
            float: 可读性指数
        """
        avg_sentence_length = self.calculate_avg_sentence_length()
        ttr = self.calculate_ttr()
        
        # 避免除零
        if ttr == 0:
            return avg_sentence_length * 0.5
        
        # 可读性指数：结合句长和词汇复杂度
        readability = 0.5 * (avg_sentence_length + 100 * (1 - ttr))
        
        return readability
    
    def get_unique_word_count(self) -> int:
        """
        获取不同词汇数量
        
        Returns:
            int: 不同词汇数
        """
        return len(set(self.tokenized_text))
    
    def to_dict(self) -> dict:
        """
        将统计结果转换为字典格式
        
        Returns:
            dict: 统计结果字典
        """
        return {
            'char_count': self.count_characters(),
            'word_count': self.count_words(),
            'sentence_count': self.count_sentences(),
            'paragraph_count': self.count_paragraphs(),
            'unique_word_count': self.get_unique_word_count(),
            'avg_sentence_length': round(self.calculate_avg_sentence_length(), 2),
            'avg_word_length': round(self.calculate_avg_word_length(), 2),
            'ttr': round(self.calculate_ttr(), 4),
            'readability_index': round(self.calculate_readability_index(), 2)
        }



class MultiDocStatistics:
    """
    多文档统计对比 - 对多个文档进行统计对比分析
    
    Attributes:
        documents: TextStatistics实例列表
        file_names: 文件名列表
    """
    
    def __init__(self, documents: List[TextStatistics], file_names: List[str]):
        """
        初始化多文档统计对比器
        
        Args:
            documents: TextStatistics实例列表
            file_names: 文件名列表
        """
        self.documents = documents
        self.file_names = file_names
    
    def compare_statistics(self) -> pd.DataFrame:
        """
        对比所有文档的统计信息
        
        Returns:
            pd.DataFrame: 统计对比表格
        """
        data = []
        
        for i, doc in enumerate(self.documents):
            file_name = self.file_names[i] if i < len(self.file_names) else f"文档{i+1}"
            stats = doc.to_dict()
            stats['file_name'] = file_name
            data.append(stats)
        
        df = pd.DataFrame(data)
        
        # 重新排列列顺序
        columns = ['file_name', 'char_count', 'word_count', 'sentence_count', 
                   'paragraph_count', 'unique_word_count', 'avg_sentence_length',
                   'avg_word_length', 'ttr', 'readability_index']
        
        # 只保留存在的列
        columns = [c for c in columns if c in df.columns]
        df = df[columns]
        
        # 重命名列为中文
        column_names = {
            'file_name': '文件名',
            'char_count': '字符数',
            'word_count': '词语数',
            'sentence_count': '句子数',
            'paragraph_count': '段落数',
            'unique_word_count': '不同词汇数',
            'avg_sentence_length': '平均句长',
            'avg_word_length': '平均词长',
            'ttr': '词汇丰富度(TTR)',
            'readability_index': '可读性指数'
        }
        df = df.rename(columns=column_names)
        
        return df
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        获取汇总统计信息（平均值、最大值、最小值）
        
        Returns:
            Dict[str, Dict[str, float]]: 汇总统计
        """
        if not self.documents:
            return {}
        
        metrics = ['char_count', 'word_count', 'sentence_count', 'paragraph_count',
                   'avg_sentence_length', 'avg_word_length', 'ttr', 'readability_index']
        
        summary = {}
        
        for metric in metrics:
            values = []
            for doc in self.documents:
                stats = doc.to_dict()
                if metric in stats:
                    values.append(stats[metric])
            
            if values:
                summary[metric] = {
                    'mean': sum(values) / len(values),
                    'max': max(values),
                    'min': min(values),
                    'count': len(values)
                }
        
        return summary
    
    def get_radar_chart_data(self, normalize: bool = True) -> Dict[str, List[float]]:
        """
        获取雷达图数据
        
        Args:
            normalize: 是否归一化数据（0-1范围）
            
        Returns:
            Dict[str, List[float]]: 雷达图数据，键为文件名，值为指标列表
        """
        # 选择用于雷达图的指标
        metrics = ['char_count', 'word_count', 'sentence_count', 
                   'avg_sentence_length', 'ttr', 'readability_index']
        
        # 收集所有文档的数据
        all_data = {}
        for i, doc in enumerate(self.documents):
            file_name = self.file_names[i] if i < len(self.file_names) else f"文档{i+1}"
            stats = doc.to_dict()
            all_data[file_name] = [stats.get(m, 0) for m in metrics]
        
        if not normalize or not all_data:
            return all_data
        
        # 归一化处理
        num_metrics = len(metrics)
        
        # 找出每个指标的最大值
        max_values = [0.0] * num_metrics
        for values in all_data.values():
            for i, v in enumerate(values):
                if v > max_values[i]:
                    max_values[i] = v
        
        # 归一化
        normalized_data = {}
        for file_name, values in all_data.items():
            normalized_values = []
            for i, v in enumerate(values):
                if max_values[i] > 0:
                    normalized_values.append(v / max_values[i])
                else:
                    normalized_values.append(0.0)
            normalized_data[file_name] = normalized_values
        
        return normalized_data
    
    def get_radar_chart_labels(self) -> List[str]:
        """
        获取雷达图标签
        
        Returns:
            List[str]: 指标标签列表
        """
        return ['字符数', '词语数', '句子数', '平均句长', '词汇丰富度', '可读性指数']
    
    def export_comparison(self) -> str:
        """
        导出对比结果为CSV格式字符串
        
        Returns:
            str: CSV格式的对比结果
        """
        df = self.compare_statistics()
        return df.to_csv(index=False, encoding='utf-8-sig')
    
    def export_comparison_to_file(self, filepath: str) -> bool:
        """
        导出对比结果到CSV文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 是否成功导出
        """
        try:
            df = self.compare_statistics()
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            return True
        except Exception:
            return False


def create_text_statistics(raw_text: str, tokenized_text: List[str]) -> TextStatistics:
    """
    创建TextStatistics实例的工厂函数
    
    Args:
        raw_text: 原始文本
        tokenized_text: 分词后的词语列表
        
    Returns:
        TextStatistics: 文本统计实例
    """
    return TextStatistics(raw_text=raw_text, tokenized_text=tokenized_text)


def create_multi_doc_statistics(raw_texts: List[str], 
                                tokenized_texts: List[List[str]],
                                file_names: List[str]) -> MultiDocStatistics:
    """
    创建MultiDocStatistics实例的工厂函数
    
    Args:
        raw_texts: 原始文本列表
        tokenized_texts: 分词后的文本列表
        file_names: 文件名列表
        
    Returns:
        MultiDocStatistics: 多文档统计实例
    """
    documents = []
    for i in range(len(raw_texts)):
        raw_text = raw_texts[i] if i < len(raw_texts) else ""
        tokenized = tokenized_texts[i] if i < len(tokenized_texts) else []
        documents.append(TextStatistics(raw_text=raw_text, tokenized_text=tokenized))
    
    return MultiDocStatistics(documents=documents, file_names=file_names)



# ============================================================================
# Streamlit UI 渲染函数
# ============================================================================

def render_text_statistics():
    """
    渲染文本统计与可读性分析模块UI
    
    Requirements: 8.4, 8.7
    """
    import streamlit as st
    import os
    from utils.session_state import log_message
    
    st.header("文本统计与可读性分析")
    
    # 功能介绍与操作手册
    with st.expander("📖 功能介绍与操作手册", expanded=False):
        st.markdown("""
        ## 📊 文本统计与可读性分析模块
        
        **功能概述**：提供文本的详细统计信息和可读性指标，帮助了解文本特征和阅读难度。
        
        ---
        
        ### 🎯 使用场景
        
        | 场景 | 关注指标 | 应用 |
        |------|----------|------|
        | 政策文本分析 | 平均句长、可读性指数 | 评估政策文本的易读性 |
        | 文本质量评估 | 词汇丰富度(TTR) | 评估文本的词汇多样性 |
        | 多文档对比 | 各项统计指标 | 比较不同文档的特征差异 |
        | 学术研究 | 全部指标 | 描述性统计、论文数据 |
        
        ---
        
        ### 📋 统计指标详解
        
        #### 基础统计指标
        
        | 指标 | 说明 | 计算方法 |
        |------|------|----------|
        | 字符数 | 文本总字符数 | 不含空白字符 |
        | 词语数 | 分词后的词语总数 | 基于jieba分词结果 |
        | 句子数 | 文本中的句子数量 | 按句号、问号、感叹号分割 |
        | 段落数 | 文本中的段落数量 | 按换行符分割 |
        | 不同词汇数 | 去重后的词语数量 | 词汇类型数(Type) |
        
        ---
        
        #### 可读性指标
        
        | 指标 | 说明 | 范围 | 解读 |
        |------|------|------|------|
        | 平均句长 | 每句平均词数 | >0 | 越大句子越长，阅读难度越高 |
        | 平均词长 | 每词平均字符数 | >0 | 中文通常1-4字 |
        | 词汇丰富度(TTR) | 不同词汇数/总词汇数 | 0-1 | 越高词汇越丰富 |
        | 可读性指数 | 综合阅读难度 | >0 | 越低越易读 |
        
        ---
        
        #### 词汇丰富度(TTR)详解
        
        **计算公式**：TTR = 不同词汇数 / 总词汇数
        
        **解读标准**：
        - > 0.7：词汇非常丰富，可能是短文本
        - 0.5-0.7：词汇较丰富
        - 0.3-0.5：词汇丰富度中等
        - < 0.3：词汇重复较多，可能是长文本或专业文本
        
        **注意**：TTR受文本长度影响，长文本TTR通常较低
        
        ---
        
        #### 可读性指数详解
        
        **计算公式**：RI = 0.5 × (平均句长 + 100 × (1 - TTR))
        
        **解读标准**：
        - < 20：非常易读
        - 20-40：较易读
        - 40-60：中等难度
        - > 60：较难读
        
        ---
        
        ### 📋 操作步骤
        
        **单文档分析**：
        1. 切换到"单文档分析"标签页
        2. 从下拉框选择要分析的文档
        3. 查看统计仪表盘和可读性评估
        4. 可展开"文本预览"查看原文
        
        **多文档对比**：
        1. 切换到"多文档对比"标签页
        2. 选择要对比的文档（至少2个）
        3. 查看统计对比表格
        4. 查看特征雷达图
        5. 查看汇总统计
        
        **导出结果**：
        1. 切换到"导出结果"标签页
        2. 点击"生成统计报告"
        3. 点击"下载CSV文件"
        4. 或指定路径导出到文件
        
        ---
        
        ### 💡 使用建议
        
        **学术研究建议**：
        - 报告基础统计指标（字符数、词语数、句子数）
        - 报告可读性指标（平均句长、TTR、可读性指数）
        - 使用多文档对比分析不同类型文本的差异
        - 导出CSV用于后续统计分析
        
        **政策分析建议**：
        - 关注可读性指数，评估政策文本的易读性
        - 比较不同时期或不同部门政策的文本特征
        - 分析政策文本的词汇丰富度变化趋势
        
        **雷达图解读**：
        - 雷达图展示归一化后的多维特征
        - 可直观比较不同文档的特征差异
        - 面积越大表示该文档在各维度上的值越高
        
        ---
        
        ### ❓ 常见问题
        
        **Q: 为什么长文本的TTR较低？**
        A: 长文本中词汇重复的概率更高，导致TTR下降。这是正常现象，比较TTR时应考虑文本长度。
        
        **Q: 可读性指数如何在论文中报告？**
        A: 可报告平均值、标准差、范围，并说明计算方法和解读标准。
        
        **Q: 如何判断文本质量？**
        A: 综合考虑多个指标：词汇丰富度反映词汇多样性，平均句长反映句子复杂度，可读性指数反映整体阅读难度。
        
        **Q: 统计结果可以用于哪些后续分析？**
        A: 可用于描述性统计、相关分析、回归分析、聚类分析等。
        """)
    
    # 检查是否已加载文件
    if not st.session_state.get("raw_texts"):
        st.warning('请先在"数据加载"选项卡中加载文件')
        return
    
    # 检查是否已完成预处理
    if not st.session_state.get("texts"):
        st.warning('请先在"文本预处理"选项卡中完成文本预处理')
        return
    
    raw_texts = st.session_state["raw_texts"]
    tokenized_texts = st.session_state["texts"]
    file_names = st.session_state.get("file_names", [])
    
    # 创建标签页
    tabs = st.tabs(["单文档分析", "多文档对比", "导出结果"])
    
    # ========== 单文档分析标签页 ==========
    with tabs[0]:
        st.subheader("单文档统计分析")
        
        # 选择文档
        selected_idx = st.selectbox(
            "选择要分析的文档",
            range(len(file_names)),
            format_func=lambda i: file_names[i] if i < len(file_names) else f"文档{i+1}",
            key="single_doc_select"
        )
        
        if selected_idx is not None:
            raw_text = raw_texts[selected_idx] if selected_idx < len(raw_texts) else ""
            tokenized = tokenized_texts[selected_idx] if selected_idx < len(tokenized_texts) else []
            
            # 创建统计实例
            stats = TextStatistics(raw_text=raw_text, tokenized_text=tokenized)
            stats_dict = stats.to_dict()
            
            # 显示仪表盘
            st.markdown("### 📊 统计仪表盘")
            
            # 第一行：基础统计
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("字符数", f"{stats_dict['char_count']:,}")
            col2.metric("词语数", f"{stats_dict['word_count']:,}")
            col3.metric("句子数", f"{stats_dict['sentence_count']:,}")
            col4.metric("段落数", f"{stats_dict['paragraph_count']:,}")
            
            # 第二行：可读性指标
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("不同词汇数", f"{stats_dict['unique_word_count']:,}")
            col2.metric("平均句长", f"{stats_dict['avg_sentence_length']:.2f} 词/句")
            col3.metric("平均词长", f"{stats_dict['avg_word_length']:.2f} 字/词")
            col4.metric("词汇丰富度(TTR)", f"{stats_dict['ttr']:.4f}")
            
            # 可读性指数
            st.markdown("### 📈 可读性评估")
            readability = stats_dict['readability_index']
            
            # 可读性等级判断
            if readability < 20:
                level = "非常易读"
                color = "green"
            elif readability < 40:
                level = "较易读"
                color = "blue"
            elif readability < 60:
                level = "中等难度"
                color = "orange"
            else:
                level = "较难读"
                color = "red"
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("可读性指数", f"{readability:.2f}")
            with col2:
                st.markdown(f"**可读性等级**: :{color}[{level}]")
                st.progress(min(readability / 100, 1.0))
            
            # 文本预览
            with st.expander("📄 文本预览", expanded=False):
                st.text_area(
                    "原始文本",
                    raw_text[:2000] + ("..." if len(raw_text) > 2000 else ""),
                    height=200,
                    disabled=True,
                    key="text_preview"
                )
    
    # ========== 多文档对比标签页 ==========
    with tabs[1]:
        st.subheader("多文档统计对比")
        
        # 选择要对比的文档
        selected_docs = st.multiselect(
            "选择要对比的文档（至少选择2个）",
            range(len(file_names)),
            format_func=lambda i: file_names[i] if i < len(file_names) else f"文档{i+1}",
            default=list(range(min(3, len(file_names)))),
            key="multi_doc_select"
        )
        
        if len(selected_docs) >= 2:
            # 创建选中文档的统计实例
            selected_raw_texts = [raw_texts[i] for i in selected_docs if i < len(raw_texts)]
            selected_tokenized = [tokenized_texts[i] for i in selected_docs if i < len(tokenized_texts)]
            selected_names = [file_names[i] for i in selected_docs if i < len(file_names)]
            
            multi_stats = create_multi_doc_statistics(
                selected_raw_texts, 
                selected_tokenized, 
                selected_names
            )
            
            # 显示对比表格
            st.markdown("### 📋 统计对比表格")
            df = multi_stats.compare_statistics()
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # 显示雷达图
            st.markdown("### 📊 特征雷达图")
            
            try:
                import plotly.graph_objects as go
                
                radar_data = multi_stats.get_radar_chart_data(normalize=True)
                labels = multi_stats.get_radar_chart_labels()
                
                fig = go.Figure()
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                
                for i, (name, values) in enumerate(radar_data.items()):
                    # 闭合雷达图
                    values_closed = values + [values[0]]
                    labels_closed = labels + [labels[0]]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values_closed,
                        theta=labels_closed,
                        fill='toself',
                        name=name,
                        line_color=colors[i % len(colors)],
                        opacity=0.6
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    showlegend=True,
                    title="文档特征对比雷达图（归一化）"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except ImportError:
                st.warning("需要安装plotly库才能显示雷达图: pip install plotly")
                st.info("已显示表格数据，雷达图功能暂不可用")
            
            # 显示汇总统计
            st.markdown("### 📈 汇总统计")
            summary = multi_stats.get_summary_statistics()
            
            summary_data = []
            metric_names = {
                'char_count': '字符数',
                'word_count': '词语数',
                'sentence_count': '句子数',
                'paragraph_count': '段落数',
                'avg_sentence_length': '平均句长',
                'avg_word_length': '平均词长',
                'ttr': '词汇丰富度(TTR)',
                'readability_index': '可读性指数'
            }
            
            for metric, values in summary.items():
                if metric in metric_names:
                    summary_data.append({
                        '指标': metric_names[metric],
                        '平均值': f"{values['mean']:.2f}",
                        '最大值': f"{values['max']:.2f}",
                        '最小值': f"{values['min']:.2f}"
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
        else:
            st.info("请至少选择2个文档进行对比分析")
    
    # ========== 导出结果标签页 ==========
    with tabs[2]:
        st.subheader("导出统计结果")
        
        # 导出所有文档的统计
        st.markdown("### 导出所有文档统计")
        
        if st.button("生成统计报告", key="generate_statistics_report"):
            # 创建所有文档的统计
            all_stats = create_multi_doc_statistics(raw_texts, tokenized_texts, file_names)
            csv_content = all_stats.export_comparison()
            
            # 保存到会话状态
            st.session_state["text_statistics_csv"] = csv_content
            st.success("统计报告已生成！")
            log_message("已生成文本统计报告")
        
        # 下载按钮
        if st.session_state.get("text_statistics_csv"):
            st.download_button(
                label="📥 下载CSV文件",
                data=st.session_state["text_statistics_csv"],
                file_name="text_statistics.csv",
                mime="text/csv",
                key="download_stats_csv"
            )
        
        # 导出到文件
        st.markdown("### 导出到指定路径")
        
        export_path = st.text_input(
            "导出路径",
            value="results/text_statistics.csv",
            key="export_path_input"
        )
        
        if st.button("导出到文件", key="export_to_file"):
            try:
                # 确保目录存在
                dir_path = os.path.dirname(export_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                
                # 创建统计并导出
                all_stats = create_multi_doc_statistics(raw_texts, tokenized_texts, file_names)
                if all_stats.export_comparison_to_file(export_path):
                    st.success(f"已成功导出到: {export_path}")
                    log_message(f"文本统计已导出到: {export_path}")
                else:
                    st.error("导出失败")
            except Exception as e:
                st.error(f"导出失败: {str(e)}")
