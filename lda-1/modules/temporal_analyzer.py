# -*- coding: utf-8 -*-
"""
时序演变分析模块 (Temporal Evolution Analysis Module)
=====================================================

本模块提供政策文本的时序演变分析功能，包括：
- 文档时间标签管理
- 关键词时序趋势分析
- 主题演变分析
- 时序数据可视化
- 结果导出

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
"""

from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import re
from datetime import datetime


@dataclass
class TimeLabel:
    """时间标签数据类"""
    doc_name: str
    time_label: str  # 时间标签（年份或日期字符串）
    sort_key: str = ""  # 用于排序的标准化键
    
    def __post_init__(self):
        """初始化后处理，生成排序键"""
        if not self.sort_key:
            self.sort_key = self._normalize_time_label(self.time_label)
    
    @staticmethod
    def _normalize_time_label(label: str) -> str:
        """
        标准化时间标签用于排序
        
        支持的格式：
        - 年份: "2020", "2021年"
        - 年月: "2020-01", "2020年1月"
        - 完整日期: "2020-01-15", "2020年1月15日"
        """
        if not label:
            return "9999"  # 无效标签排到最后
        
        # 移除中文字符，提取数字
        cleaned = re.sub(r'[年月日]', '-', label)
        cleaned = re.sub(r'-+', '-', cleaned).strip('-')
        
        # 尝试解析不同格式
        parts = cleaned.split('-')
        
        # 补齐为标准格式 YYYY-MM-DD
        if len(parts) == 1:
            # 只有年份
            return f"{parts[0]:0>4}-00-00"
        elif len(parts) == 2:
            # 年月
            return f"{parts[0]:0>4}-{parts[1]:0>2}-00"
        elif len(parts) >= 3:
            # 完整日期
            return f"{parts[0]:0>4}-{parts[1]:0>2}-{parts[2]:0>2}"
        
        return label


class TemporalAnalyzer:
    """
    时序分析器 - 分析文本随时间的演变趋势
    
    Attributes:
        texts: 分词后的文本列表（每个文本是词语列表）
        file_names: 文档名称列表
        time_labels: 文档时间标签映射
    
    Requirements: 4.1, 4.2, 4.3, 4.5
    """
    
    def __init__(self, texts: List[List[str]], file_names: List[str]):
        """
        初始化时序分析器
        
        Args:
            texts: 分词后的文本列表
            file_names: 文档名称列表
        """
        self.texts = texts if texts else []
        self.file_names = file_names if file_names else []
        self.time_labels: Dict[str, TimeLabel] = {}
        self._sorted_periods: Optional[List[str]] = None
    
    def set_time_label(self, doc_name: str, time_label: str) -> bool:
        """
        为文档设置时间标签
        
        Args:
            doc_name: 文档名称
            time_label: 时间标签（年份或日期）
        
        Returns:
            bool: 是否设置成功
        
        Requirements: 4.1
        """
        if doc_name not in self.file_names:
            return False
        
        if not time_label or not time_label.strip():
            # 移除空标签
            if doc_name in self.time_labels:
                del self.time_labels[doc_name]
            return True
        
        self.time_labels[doc_name] = TimeLabel(doc_name, time_label.strip())
        self._sorted_periods = None  # 清除缓存
        return True
    
    def set_time_labels_batch(self, labels: Dict[str, str]) -> int:
        """
        批量设置时间标签
        
        Args:
            labels: 文档名到时间标签的映射
        
        Returns:
            int: 成功设置的数量
        """
        count = 0
        for doc_name, time_label in labels.items():
            if self.set_time_label(doc_name, time_label):
                count += 1
        return count
    
    def get_time_label(self, doc_name: str) -> Optional[str]:
        """
        获取文档的时间标签
        
        Args:
            doc_name: 文档名称
        
        Returns:
            Optional[str]: 时间标签，如果未设置则返回None
        """
        label = self.time_labels.get(doc_name)
        return label.time_label if label else None
    
    def get_all_time_labels(self) -> Dict[str, str]:
        """
        获取所有文档的时间标签
        
        Returns:
            Dict[str, str]: 文档名到时间标签的映射
        """
        return {name: label.time_label for name, label in self.time_labels.items()}
    
    def get_labeled_documents_count(self) -> int:
        """获取已标注时间标签的文档数量"""
        return len(self.time_labels)
    
    def get_unlabeled_documents(self) -> List[str]:
        """获取未标注时间标签的文档列表"""
        return [name for name in self.file_names if name not in self.time_labels]
    
    def get_sorted_periods(self) -> List[str]:
        """
        获取按时间排序的时间段列表
        
        Returns:
            List[str]: 排序后的时间标签列表（去重）
        
        Requirements: 4.2
        """
        if self._sorted_periods is not None:
            return self._sorted_periods
        
        # 收集所有唯一的时间标签
        unique_labels = {}
        for label in self.time_labels.values():
            if label.time_label not in unique_labels:
                unique_labels[label.time_label] = label.sort_key
        
        # 按排序键排序
        sorted_labels = sorted(unique_labels.items(), key=lambda x: x[1])
        self._sorted_periods = [label for label, _ in sorted_labels]
        
        return self._sorted_periods
    
    def get_documents_by_period(self, period: str) -> List[str]:
        """
        获取指定时间段的文档列表
        
        Args:
            period: 时间标签
        
        Returns:
            List[str]: 该时间段的文档名称列表
        
        Requirements: 4.2
        """
        return [
            name for name, label in self.time_labels.items()
            if label.time_label == period
        ]
    
    def get_documents_sorted_by_time(self) -> List[Tuple[str, str]]:
        """
        获取按时间排序的文档列表
        
        Returns:
            List[Tuple[str, str]]: (文档名, 时间标签) 列表，按时间升序排列
        
        Requirements: 4.2
        """
        labeled_docs = [
            (name, label.time_label, label.sort_key)
            for name, label in self.time_labels.items()
        ]
        
        # 按排序键排序
        labeled_docs.sort(key=lambda x: x[2])
        
        return [(name, time_label) for name, time_label, _ in labeled_docs]
    
    def _get_doc_index(self, doc_name: str) -> int:
        """获取文档在列表中的索引"""
        try:
            return self.file_names.index(doc_name)
        except ValueError:
            return -1
    
    def analyze_keyword_trend(self, keyword: str) -> Dict[str, int]:
        """
        分析单个关键词在不同时间段的频率变化
        
        Args:
            keyword: 要分析的关键词
        
        Returns:
            Dict[str, int]: 时间段 -> 频率 的映射，按时间排序
        
        Requirements: 4.3
        """
        if not self.time_labels:
            return {}
        
        # 按时间段统计关键词频率
        period_freq = defaultdict(int)
        
        for doc_name, label in self.time_labels.items():
            doc_idx = self._get_doc_index(doc_name)
            if doc_idx >= 0 and doc_idx < len(self.texts):
                # 统计该文档中关键词出现次数
                count = self.texts[doc_idx].count(keyword)
                period_freq[label.time_label] += count
        
        # 按时间排序
        sorted_periods = self.get_sorted_periods()
        result = {}
        for period in sorted_periods:
            result[period] = period_freq.get(period, 0)
        
        return result
    
    def analyze_keywords_trends(self, keywords: List[str]) -> Dict[str, Dict[str, int]]:
        """
        分析多个关键词的时序趋势
        
        Args:
            keywords: 关键词列表
        
        Returns:
            Dict[str, Dict[str, int]]: 关键词 -> (时间段 -> 频率) 的映射
        
        Requirements: 4.3
        """
        return {keyword: self.analyze_keyword_trend(keyword) for keyword in keywords}
    
    def get_period_word_frequency(self, period: str, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        获取指定时间段的高频词
        
        Args:
            period: 时间标签
            top_n: 返回的词语数量
        
        Returns:
            List[Tuple[str, int]]: (词语, 频率) 列表
        """
        docs = self.get_documents_by_period(period)
        
        word_counter = Counter()
        for doc_name in docs:
            doc_idx = self._get_doc_index(doc_name)
            if doc_idx >= 0 and doc_idx < len(self.texts):
                word_counter.update(self.texts[doc_idx])
        
        return word_counter.most_common(top_n)
    
    def analyze_topic_evolution(self, doc_topic_dist: np.ndarray) -> Dict[str, List[float]]:
        """
        分析主题在不同时间段的分布变化
        
        Args:
            doc_topic_dist: 文档-主题分布矩阵，形状为 (n_docs, n_topics)
        
        Returns:
            Dict[str, List[float]]: 时间段 -> 各主题平均概率列表
        
        Requirements: 4.5
        """
        if doc_topic_dist is None or len(doc_topic_dist) == 0:
            return {}
        
        if not self.time_labels:
            return {}
        
        n_topics = doc_topic_dist.shape[1] if len(doc_topic_dist.shape) > 1 else 1
        sorted_periods = self.get_sorted_periods()
        
        result = {}
        for period in sorted_periods:
            docs = self.get_documents_by_period(period)
            
            if not docs:
                result[period] = [0.0] * n_topics
                continue
            
            # 收集该时间段所有文档的主题分布
            topic_sums = np.zeros(n_topics)
            doc_count = 0
            
            for doc_name in docs:
                doc_idx = self._get_doc_index(doc_name)
                if doc_idx >= 0 and doc_idx < len(doc_topic_dist):
                    topic_sums += doc_topic_dist[doc_idx]
                    doc_count += 1
            
            # 计算平均值
            if doc_count > 0:
                result[period] = (topic_sums / doc_count).tolist()
            else:
                result[period] = [0.0] * n_topics
        
        return result
    
    def get_emerging_keywords(self, recent_periods: int = 2, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        识别新兴关键词（在最近时间段频率显著增加的词）
        
        Args:
            recent_periods: 最近的时间段数量
            top_n: 返回的关键词数量
        
        Returns:
            List[Tuple[str, float]]: (关键词, 增长率) 列表
        """
        sorted_periods = self.get_sorted_periods()
        
        if len(sorted_periods) < 2:
            return []
        
        # 分割为早期和近期
        split_point = max(1, len(sorted_periods) - recent_periods)
        early_periods = sorted_periods[:split_point]
        recent_periods_list = sorted_periods[split_point:]
        
        # 统计早期词频
        early_freq = Counter()
        early_doc_count = 0
        for period in early_periods:
            docs = self.get_documents_by_period(period)
            early_doc_count += len(docs)
            for doc_name in docs:
                doc_idx = self._get_doc_index(doc_name)
                if doc_idx >= 0 and doc_idx < len(self.texts):
                    early_freq.update(self.texts[doc_idx])
        
        # 统计近期词频
        recent_freq = Counter()
        recent_doc_count = 0
        for period in recent_periods_list:
            docs = self.get_documents_by_period(period)
            recent_doc_count += len(docs)
            for doc_name in docs:
                doc_idx = self._get_doc_index(doc_name)
                if doc_idx >= 0 and doc_idx < len(self.texts):
                    recent_freq.update(self.texts[doc_idx])
        
        if early_doc_count == 0 or recent_doc_count == 0:
            return []
        
        # 计算增长率
        growth_rates = []
        for word, recent_count in recent_freq.items():
            early_count = early_freq.get(word, 0)
            
            # 归一化频率
            early_rate = early_count / early_doc_count if early_doc_count > 0 else 0
            recent_rate = recent_count / recent_doc_count if recent_doc_count > 0 else 0
            
            # 计算增长率（避免除零）
            if early_rate > 0:
                growth = (recent_rate - early_rate) / early_rate
            elif recent_rate > 0:
                growth = float('inf')  # 新出现的词
            else:
                growth = 0
            
            if growth > 0 and recent_count >= 2:  # 只保留增长的词
                growth_rates.append((word, growth))
        
        # 按增长率排序
        growth_rates.sort(key=lambda x: -x[1])
        
        # 处理无穷大值
        result = []
        for word, rate in growth_rates[:top_n]:
            if rate == float('inf'):
                result.append((word, 999.99))  # 用大数表示新词
            else:
                result.append((word, round(rate * 100, 2)))  # 转为百分比
        
        return result
    
    def get_declining_keywords(self, recent_periods: int = 2, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        识别衰退关键词（在最近时间段频率显著下降的词）
        
        Args:
            recent_periods: 最近的时间段数量
            top_n: 返回的关键词数量
        
        Returns:
            List[Tuple[str, float]]: (关键词, 下降率) 列表
        """
        sorted_periods = self.get_sorted_periods()
        
        if len(sorted_periods) < 2:
            return []
        
        # 分割为早期和近期
        split_point = max(1, len(sorted_periods) - recent_periods)
        early_periods = sorted_periods[:split_point]
        recent_periods_list = sorted_periods[split_point:]
        
        # 统计早期词频
        early_freq = Counter()
        early_doc_count = 0
        for period in early_periods:
            docs = self.get_documents_by_period(period)
            early_doc_count += len(docs)
            for doc_name in docs:
                doc_idx = self._get_doc_index(doc_name)
                if doc_idx >= 0 and doc_idx < len(self.texts):
                    early_freq.update(self.texts[doc_idx])
        
        # 统计近期词频
        recent_freq = Counter()
        recent_doc_count = 0
        for period in recent_periods_list:
            docs = self.get_documents_by_period(period)
            recent_doc_count += len(docs)
            for doc_name in docs:
                doc_idx = self._get_doc_index(doc_name)
                if doc_idx >= 0 and doc_idx < len(self.texts):
                    recent_freq.update(self.texts[doc_idx])
        
        if early_doc_count == 0 or recent_doc_count == 0:
            return []
        
        # 计算下降率
        decline_rates = []
        for word, early_count in early_freq.items():
            recent_count = recent_freq.get(word, 0)
            
            # 归一化频率
            early_rate = early_count / early_doc_count if early_doc_count > 0 else 0
            recent_rate = recent_count / recent_doc_count if recent_doc_count > 0 else 0
            
            # 计算下降率
            if early_rate > 0:
                decline = (early_rate - recent_rate) / early_rate
            else:
                decline = 0
            
            if decline > 0 and early_count >= 2:  # 只保留下降的词
                decline_rates.append((word, decline))
        
        # 按下降率排序
        decline_rates.sort(key=lambda x: -x[1])
        
        return [(word, round(rate * 100, 2)) for word, rate in decline_rates[:top_n]]
    
    def export_trend_data(self, keywords: Optional[List[str]] = None) -> str:
        """
        导出关键词趋势数据为CSV格式
        
        Args:
            keywords: 要导出的关键词列表，如果为None则导出所有高频词
        
        Returns:
            str: CSV格式字符串
        
        Requirements: 4.7
        """
        if not self.time_labels:
            return ""
        
        sorted_periods = self.get_sorted_periods()
        
        # 如果没有指定关键词，获取所有时间段的高频词
        if keywords is None:
            all_words = Counter()
            for doc_idx, text in enumerate(self.texts):
                if self.file_names[doc_idx] in self.time_labels:
                    all_words.update(text)
            keywords = [word for word, _ in all_words.most_common(50)]
        
        # 构建数据
        data = []
        for keyword in keywords:
            trend = self.analyze_keyword_trend(keyword)
            row = {"关键词": keyword}
            for period in sorted_periods:
                row[period] = trend.get(period, 0)
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='utf-8-sig')
    
    def export_topic_evolution_data(self, doc_topic_dist: np.ndarray, topic_names: Optional[List[str]] = None) -> str:
        """
        导出主题演变数据为CSV格式
        
        Args:
            doc_topic_dist: 文档-主题分布矩阵
            topic_names: 主题名称列表
        
        Returns:
            str: CSV格式字符串
        
        Requirements: 4.7
        """
        evolution = self.analyze_topic_evolution(doc_topic_dist)
        
        if not evolution:
            return ""
        
        n_topics = len(list(evolution.values())[0])
        
        if topic_names is None:
            topic_names = [f"主题{i+1}" for i in range(n_topics)]
        
        # 构建数据
        data = []
        for period, topic_probs in evolution.items():
            row = {"时间段": period}
            for i, prob in enumerate(topic_probs):
                topic_name = topic_names[i] if i < len(topic_names) else f"主题{i+1}"
                row[topic_name] = round(prob, 4)
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='utf-8-sig')
    
    def export_time_labels(self) -> str:
        """
        导出时间标签数据为CSV格式
        
        Returns:
            str: CSV格式字符串
        """
        data = []
        for doc_name in self.file_names:
            label = self.time_labels.get(doc_name)
            data.append({
                "文档名": doc_name,
                "时间标签": label.time_label if label else ""
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='utf-8-sig')
    
    def import_time_labels(self, csv_content: str) -> int:
        """
        从CSV导入时间标签
        
        Args:
            csv_content: CSV格式字符串
        
        Returns:
            int: 成功导入的数量
        """
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content))
            
            count = 0
            for _, row in df.iterrows():
                doc_name = str(row.get("文档名", ""))
                time_label = str(row.get("时间标签", ""))
                
                if doc_name and time_label and self.set_time_label(doc_name, time_label):
                    count += 1
            
            return count
        except Exception:
            return 0
    
    def auto_extract_time_from_filename(self, pattern: str = r'(\d{4})') -> int:
        """
        从文件名自动提取时间标签
        
        Args:
            pattern: 正则表达式模式，默认提取4位数字（年份）
        
        Returns:
            int: 成功提取的数量
        """
        count = 0
        regex = re.compile(pattern)
        
        for doc_name in self.file_names:
            match = regex.search(doc_name)
            if match:
                time_label = match.group(1)
                if self.set_time_label(doc_name, time_label):
                    count += 1
        
        return count



# ============================================================================
# Streamlit UI 渲染函数
# ============================================================================

def render_temporal_analyzer():
    """
    渲染时序演变分析模块UI
    
    Requirements: 4.4, 4.6, 4.7
    """
    import streamlit as st
    from utils.session_state import log_message
    
    st.header("📅 时序演变分析")
    
    # 功能介绍与操作手册
    with st.expander("📖 功能介绍与操作手册", expanded=False):
        st.markdown("""
        ## 📅 时序演变分析模块
        
        **功能概述**：分析政策文本随时间的演变趋势，追踪关键词和主题的变化。
        
        ---
        
        ### 🎯 使用场景
        
        | 场景 | 关注点 | 应用 |
        |------|--------|------|
        | 政策演变研究 | 关键词趋势 | 追踪政策议题的发展变化 |
        | 主题变迁分析 | 主题分布变化 | 了解政策重心的转移 |
        | 新兴议题发现 | 新兴关键词 | 识别新出现的政策热点 |
        | 衰退议题识别 | 衰退关键词 | 发现逐渐淡出的政策议题 |
        
        ---
        
        ### 📋 操作步骤
        
        **1. 设置时间标签**：
        - 手动为每个文档设置时间标签（年份或日期）
        - 或使用"从文件名提取"功能自动提取
        - 或导入已有的时间标签CSV文件
        
        **2. 关键词趋势分析**：
        - 输入要追踪的关键词（逗号分隔）
        - 查看关键词在不同时间段的频率变化
        - 折线图展示趋势变化
        
        **3. 主题演变分析**（需先完成主题建模）：
        - 查看各主题在不同时间段的分布变化
        - 堆叠面积图展示主题演变
        
        **4. 导出分析结果**：
        - 导出关键词趋势数据
        - 导出主题演变数据
        - 导出时间标签设置
        
        ---
        
        ### ⚙️ 时间标签格式
        
        | 格式 | 示例 | 说明 |
        |------|------|------|
        | 年份 | 2020, 2021年 | 按年度分析 |
        | 年月 | 2020-01, 2020年1月 | 按月度分析 |
        | 完整日期 | 2020-01-15 | 精确到日 |
        
        ---
        
        ### 💡 使用建议
        
        **学术研究建议**：
        - 确保时间标签覆盖所有文档
        - 选择有代表性的关键词进行趋势分析
        - 结合主题建模结果进行综合分析
        
        **可视化建议**：
        - 关键词数量建议3-8个，便于图表清晰展示
        - 时间跨度较长时，可按年度聚合分析
        """)
    
    # 检查数据
    if not st.session_state.get("texts"):
        st.warning("⚠️ 请先在「文本预处理」标签页中完成文本预处理")
        return
    
    texts = st.session_state["texts"]
    file_names = st.session_state.get("file_names", [])
    
    if not file_names:
        st.warning("⚠️ 未找到文档名称列表")
        return
    
    # 获取或创建分析器
    if "temporal_analyzer" not in st.session_state or st.session_state["temporal_analyzer"] is None:
        st.session_state["temporal_analyzer"] = TemporalAnalyzer(texts, file_names)
        # 恢复已保存的时间标签
        if st.session_state.get("time_labels"):
            st.session_state["temporal_analyzer"].set_time_labels_batch(st.session_state["time_labels"])
    
    analyzer = st.session_state["temporal_analyzer"]
    
    # 创建标签页
    tabs = st.tabs([
        "🏷️ 时间标签设置",
        "📈 关键词趋势",
        "📊 主题演变",
        "🔍 新兴/衰退词",
        "💾 导出"
    ])
    
    # ========== 时间标签设置 ==========
    with tabs[0]:
        st.subheader("时间标签设置")
        
        # 统计信息
        labeled_count = analyzer.get_labeled_documents_count()
        total_count = len(file_names)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("总文档数", total_count)
        col2.metric("已标注", labeled_count)
        col3.metric("未标注", total_count - labeled_count)
        
        st.markdown("---")
        
        # 自动提取选项
        st.markdown("**🔧 自动提取时间标签**")
        auto_col1, auto_col2 = st.columns([3, 1])
        
        with auto_col1:
            extract_pattern = st.text_input(
                "正则表达式模式",
                value=r'(\d{4})',
                help="默认提取4位数字作为年份，可自定义模式"
            )
        
        with auto_col2:
            st.write("")  # 占位
            st.write("")
            if st.button("从文件名提取", type="secondary"):
                count = analyzer.auto_extract_time_from_filename(extract_pattern)
                if count > 0:
                    st.success(f"成功提取 {count} 个时间标签")
                    # 保存到会话状态
                    st.session_state["time_labels"] = analyzer.get_all_time_labels()
                    log_message(f"从文件名自动提取了 {count} 个时间标签")
                    st.rerun()
                else:
                    st.warning("未能从文件名中提取时间标签")
        
        st.markdown("---")
        
        # 手动设置时间标签
        st.markdown("**✏️ 手动设置时间标签**")
        
        # 使用数据编辑器
        current_labels = analyzer.get_all_time_labels()
        label_data = []
        for name in file_names:
            label_data.append({
                "文档名": name,
                "时间标签": current_labels.get(name, "")
            })
        
        df = pd.DataFrame(label_data)
        
        edited_df = st.data_editor(
            df,
            column_config={
                "文档名": st.column_config.TextColumn("文档名", disabled=True),
                "时间标签": st.column_config.TextColumn(
                    "时间标签",
                    help="输入年份（如2020）或日期（如2020-01-15）"
                )
            },
            use_container_width=True,
            hide_index=True,
            num_rows="fixed"
        )
        
        # 保存按钮
        if st.button("💾 保存时间标签", type="primary"):
            # 更新分析器中的时间标签
            for _, row in edited_df.iterrows():
                doc_name = row["文档名"]
                time_label = row["时间标签"]
                analyzer.set_time_label(doc_name, time_label if pd.notna(time_label) else "")
            
            # 保存到会话状态
            st.session_state["time_labels"] = analyzer.get_all_time_labels()
            st.success("时间标签已保存")
            log_message(f"保存了 {analyzer.get_labeled_documents_count()} 个时间标签")
        
        # 导入/导出时间标签
        st.markdown("---")
        st.markdown("**📁 导入/导出时间标签**")
        
        imp_col, exp_col = st.columns(2)
        
        with imp_col:
            uploaded_file = st.file_uploader(
                "导入时间标签CSV",
                type=["csv"],
                help="CSV文件需包含'文档名'和'时间标签'两列"
            )
            if uploaded_file is not None:
                content = uploaded_file.read().decode('utf-8-sig')
                count = analyzer.import_time_labels(content)
                if count > 0:
                    st.session_state["time_labels"] = analyzer.get_all_time_labels()
                    st.success(f"成功导入 {count} 个时间标签")
                    st.rerun()
                else:
                    st.error("导入失败，请检查CSV格式")
        
        with exp_col:
            csv_content = analyzer.export_time_labels()
            st.download_button(
                label="📥 导出时间标签CSV",
                data=csv_content,
                file_name="time_labels.csv",
                mime="text/csv"
            )
    
    # ========== 关键词趋势分析 ==========
    with tabs[1]:
        st.subheader("关键词趋势分析")
        
        if analyzer.get_labeled_documents_count() < 2:
            st.warning("⚠️ 请先在「时间标签设置」中为至少2个文档设置时间标签")
            return
        
        # 显示时间段信息
        sorted_periods = analyzer.get_sorted_periods()
        st.info(f"📅 时间范围: {sorted_periods[0]} ~ {sorted_periods[-1]} ({len(sorted_periods)} 个时间段)")
        
        # 关键词输入
        keywords_input = st.text_input(
            "输入要追踪的关键词（逗号分隔）",
            value=st.session_state.get("temporal_keywords", ""),
            placeholder="例如: 创新, 发展, 改革, 数字化"
        )
        
        if st.button("📈 分析关键词趋势", type="primary"):
            if not keywords_input.strip():
                st.warning("请输入至少一个关键词")
            else:
                keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
                st.session_state["temporal_keywords"] = keywords_input
                
                with st.spinner("正在分析关键词趋势..."):
                    trends = analyzer.analyze_keywords_trends(keywords)
                    st.session_state["keyword_trends"] = trends
                    log_message(f"分析了 {len(keywords)} 个关键词的时序趋势")
        
        # 显示趋势图
        if st.session_state.get("keyword_trends"):
            trends = st.session_state["keyword_trends"]
            
            # 准备数据
            chart_data = []
            for keyword, trend in trends.items():
                for period, freq in trend.items():
                    chart_data.append({
                        "时间段": period,
                        "关键词": keyword,
                        "频率": freq
                    })
            
            if chart_data:
                df = pd.DataFrame(chart_data)
                
                # 使用Plotly绘制折线图
                try:
                    import plotly.express as px
                    
                    fig = px.line(
                        df,
                        x="时间段",
                        y="频率",
                        color="关键词",
                        markers=True,
                        title="关键词时序趋势"
                    )
                    fig.update_layout(
                        xaxis_title="时间段",
                        yaxis_title="出现频率",
                        legend_title="关键词",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except ImportError:
                    # 使用Streamlit原生图表
                    pivot_df = df.pivot(index="时间段", columns="关键词", values="频率")
                    st.line_chart(pivot_df)
                
                # 显示数据表格
                with st.expander("📊 查看详细数据"):
                    pivot_df = df.pivot(index="关键词", columns="时间段", values="频率").fillna(0)
                    st.dataframe(pivot_df, use_container_width=True)
    
    # ========== 主题演变分析 ==========
    with tabs[2]:
        st.subheader("主题演变分析")
        
        if analyzer.get_labeled_documents_count() < 2:
            st.warning("⚠️ 请先在「时间标签设置」中为至少2个文档设置时间标签")
            return
        
        # 检查是否有主题模型结果
        doc_topic_dist = st.session_state.get("doc_topic_dist")
        
        if doc_topic_dist is None:
            st.warning("⚠️ 请先在「主题建模」标签页中完成LDA主题建模")
            return
        
        # 获取主题关键词作为主题名称
        topic_keywords = st.session_state.get("topic_keywords", {})
        n_topics = doc_topic_dist.shape[1] if len(doc_topic_dist.shape) > 1 else 1
        
        topic_names = []
        for i in range(n_topics):
            if i in topic_keywords and topic_keywords[i]:
                # 取前3个关键词作为主题名称
                keywords = topic_keywords[i][:3] if isinstance(topic_keywords[i], list) else []
                name = f"主题{i+1}: {', '.join(keywords)}" if keywords else f"主题{i+1}"
            else:
                name = f"主题{i+1}"
            topic_names.append(name)
        
        if st.button("📊 分析主题演变", type="primary"):
            with st.spinner("正在分析主题演变..."):
                evolution = analyzer.analyze_topic_evolution(doc_topic_dist)
                st.session_state["topic_evolution"] = evolution
                log_message("完成主题演变分析")
        
        # 显示主题演变图
        if st.session_state.get("topic_evolution"):
            evolution = st.session_state["topic_evolution"]
            
            # 准备数据
            chart_data = []
            for period, probs in evolution.items():
                for i, prob in enumerate(probs):
                    chart_data.append({
                        "时间段": period,
                        "主题": topic_names[i] if i < len(topic_names) else f"主题{i+1}",
                        "概率": prob
                    })
            
            if chart_data:
                df = pd.DataFrame(chart_data)
                
                try:
                    import plotly.express as px
                    
                    # 堆叠面积图
                    fig = px.area(
                        df,
                        x="时间段",
                        y="概率",
                        color="主题",
                        title="主题时序演变（堆叠面积图）"
                    )
                    fig.update_layout(
                        xaxis_title="时间段",
                        yaxis_title="主题概率",
                        legend_title="主题",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 折线图
                    fig2 = px.line(
                        df,
                        x="时间段",
                        y="概率",
                        color="主题",
                        markers=True,
                        title="主题时序演变（折线图）"
                    )
                    fig2.update_layout(
                        xaxis_title="时间段",
                        yaxis_title="主题概率",
                        legend_title="主题"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
                except ImportError:
                    pivot_df = df.pivot(index="时间段", columns="主题", values="概率")
                    st.area_chart(pivot_df)
                
                # 显示数据表格
                with st.expander("📊 查看详细数据"):
                    pivot_df = df.pivot(index="时间段", columns="主题", values="概率").round(4)
                    st.dataframe(pivot_df, use_container_width=True)
    
    # ========== 新兴/衰退关键词 ==========
    with tabs[3]:
        st.subheader("新兴与衰退关键词")
        
        if analyzer.get_labeled_documents_count() < 2:
            st.warning("⚠️ 请先在「时间标签设置」中为至少2个文档设置时间标签")
            return
        
        sorted_periods = analyzer.get_sorted_periods()
        if len(sorted_periods) < 2:
            st.warning("⚠️ 需要至少2个不同的时间段才能进行趋势分析")
            return
        
        # 参数设置
        col1, col2 = st.columns(2)
        with col1:
            recent_n = st.slider(
                "近期时间段数量",
                min_value=1,
                max_value=max(1, len(sorted_periods) - 1),
                value=min(2, len(sorted_periods) - 1),
                help="将最近N个时间段与之前的时间段进行对比"
            )
        with col2:
            top_n = st.slider(
                "显示关键词数量",
                min_value=5,
                max_value=30,
                value=10
            )
        
        if st.button("🔍 分析新兴/衰退关键词", type="primary"):
            with st.spinner("正在分析..."):
                emerging = analyzer.get_emerging_keywords(recent_n, top_n)
                declining = analyzer.get_declining_keywords(recent_n, top_n)
                
                st.session_state["emerging_keywords"] = emerging
                st.session_state["declining_keywords"] = declining
                log_message("完成新兴/衰退关键词分析")
        
        # 显示结果
        if st.session_state.get("emerging_keywords") or st.session_state.get("declining_keywords"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 新兴关键词")
                st.caption("在近期时间段频率显著增加的词")
                
                emerging = st.session_state.get("emerging_keywords", [])
                if emerging:
                    df = pd.DataFrame(emerging, columns=["关键词", "增长率(%)"])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("未发现显著增长的关键词")
            
            with col2:
                st.markdown("### 📉 衰退关键词")
                st.caption("在近期时间段频率显著下降的词")
                
                declining = st.session_state.get("declining_keywords", [])
                if declining:
                    df = pd.DataFrame(declining, columns=["关键词", "下降率(%)"])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("未发现显著下降的关键词")
    
    # ========== 导出 ==========
    with tabs[4]:
        st.subheader("导出分析结果")
        
        if analyzer.get_labeled_documents_count() == 0:
            st.warning("⚠️ 请先设置时间标签")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 关键词趋势数据**")
            if st.session_state.get("keyword_trends"):
                keywords = list(st.session_state["keyword_trends"].keys())
                csv_content = analyzer.export_trend_data(keywords)
                st.download_button(
                    label="📥 下载关键词趋势CSV",
                    data=csv_content,
                    file_name="keyword_trends.csv",
                    mime="text/csv"
                )
            else:
                st.info("请先进行关键词趋势分析")
        
        with col2:
            st.markdown("**📈 主题演变数据**")
            doc_topic_dist = st.session_state.get("doc_topic_dist")
            if doc_topic_dist is not None and st.session_state.get("topic_evolution"):
                topic_keywords = st.session_state.get("topic_keywords", {})
                n_topics = doc_topic_dist.shape[1]
                topic_names = [f"主题{i+1}" for i in range(n_topics)]
                
                csv_content = analyzer.export_topic_evolution_data(doc_topic_dist, topic_names)
                st.download_button(
                    label="📥 下载主题演变CSV",
                    data=csv_content,
                    file_name="topic_evolution.csv",
                    mime="text/csv"
                )
            else:
                st.info("请先进行主题演变分析")
        
        st.markdown("---")
        
        # 时间标签导出
        st.markdown("**🏷️ 时间标签数据**")
        csv_content = analyzer.export_time_labels()
        st.download_button(
            label="📥 下载时间标签CSV",
            data=csv_content,
            file_name="time_labels.csv",
            mime="text/csv"
        )
