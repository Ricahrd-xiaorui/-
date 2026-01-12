# -*- coding: utf-8 -*-
"""
引用与参考分析模块 (Citation Analysis Module)
=============================================

本模块提供政策文本间的引用关系分析功能，包括：
- 引用提取（识别文本中对其他政策文件的引用）
- 引用网络构建
- 核心文档识别
- 引用关系可视化
- 结果导出

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import re


@dataclass
class Citation:
    """引用数据类"""
    source_doc: str  # 引用来源文档
    cited_text: str  # 被引用的文本/文件名
    context: str = ""  # 引用上下文
    position: int = 0  # 在文档中的位置


@dataclass
class CitationStats:
    """引用统计数据类"""
    doc_name: str
    cited_by_count: int  # 被引用次数
    cites_count: int  # 引用其他文档次数
    cited_by: List[str] = field(default_factory=list)  # 被哪些文档引用
    cites: List[str] = field(default_factory=list)  # 引用了哪些文档


class CitationAnalyzer:
    """
    引用分析器 - 分析政策文本间的引用关系
    
    Attributes:
        raw_texts: 原始文本列表
        file_names: 文档名称列表
        citations: 提取的引用列表
    
    Requirements: 6.1, 6.2, 6.3, 6.5, 6.6
    """
    
    # 中文政策文件引用模式
    CITATION_PATTERNS = [
        # 《xxx》格式
        r'《([^》]+)》',
        # 根据xxx规定
        r'根据[「「]?([^」」，。、\s]+)[」」]?(?:的)?(?:规定|要求|精神)',
        # 依据xxx
        r'依据[「「]?([^」」，。、\s]+)[」」]?',
        # 按照xxx
        r'按照[「「]?([^」」，。、\s]+)[」」]?(?:的)?(?:规定|要求)',
        # 参照xxx
        r'参照[「「]?([^」」，。、\s]+)[」」]?',
        # 贯彻xxx
        r'贯彻[「「]?([^」」，。、\s]+)[」」]?',
        # 落实xxx
        r'落实[「「]?([^」」，。、\s]+)[」」]?',
        # xxx号文
        r'([^\s，。、]+(?:号|字)[^\s，。、]*文)',
        # xxx通知/意见/办法等
        r'[「「]([^」」]+(?:通知|意见|办法|规定|条例|法|决定|方案|纲要|规划))[」」]',
    ]
    
    def __init__(self, raw_texts: List[str], file_names: List[str]):
        """
        初始化引用分析器
        
        Args:
            raw_texts: 原始文本列表
            file_names: 文档名称列表
        """
        self.raw_texts = raw_texts if raw_texts else []
        self.file_names = file_names if file_names else []
        self.citations: List[Citation] = []
        self._citation_network: Optional[Dict[str, List[str]]] = None
        self._compiled_patterns = [re.compile(p) for p in self.CITATION_PATTERNS]
    
    def extract_citations(self) -> Dict[str, List[str]]:
        """
        从所有文档中提取引用
        
        Returns:
            Dict[str, List[str]]: 文档名 -> 被引用文件名列表
        
        Requirements: 6.1, 6.2
        """
        self.citations = []
        citation_map: Dict[str, List[str]] = defaultdict(list)
        
        for doc_idx, raw_text in enumerate(self.raw_texts):
            if doc_idx >= len(self.file_names):
                continue
            
            doc_name = self.file_names[doc_idx]
            cited_texts = self._extract_citations_from_text(raw_text, doc_name)
            
            for citation in cited_texts:
                self.citations.append(citation)
                if citation.cited_text not in citation_map[doc_name]:
                    citation_map[doc_name].append(citation.cited_text)
        
        self._citation_network = dict(citation_map)
        return self._citation_network
    
    def _extract_citations_from_text(self, text: str, doc_name: str) -> List[Citation]:
        """
        从单个文本中提取引用
        
        Args:
            text: 原始文本
            doc_name: 文档名称
        
        Returns:
            List[Citation]: 引用列表
        """
        if not text:
            return []
        
        citations = []
        seen_citations: Set[str] = set()
        
        for pattern in self._compiled_patterns:
            for match in pattern.finditer(text):
                cited_text = match.group(1).strip()
                
                # 清理引用文本（移除多余的标点和空白）
                cited_text = self._clean_citation_text(cited_text)
                
                # 过滤无效引用
                if not self._is_valid_citation(cited_text):
                    continue
                
                # 去重（使用标准化后的文本）
                normalized = self._normalize_citation(cited_text)
                if normalized in seen_citations:
                    continue
                seen_citations.add(normalized)
                
                # 提取上下文（前后各50个字符）
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                citations.append(Citation(
                    source_doc=doc_name,
                    cited_text=cited_text,
                    context=context,
                    position=match.start()
                ))
        
        return citations
    
    def _clean_citation_text(self, text: str) -> str:
        """
        清理引用文本
        
        Args:
            text: 原始引用文本
        
        Returns:
            str: 清理后的文本
        """
        if not text:
            return ""
        
        # 移除首尾的书名号
        text = re.sub(r'^[《「「\s]+', '', text)
        text = re.sub(r'[》」」\s]+$', '', text)
        
        # 移除末尾的"的"、"精神"等后缀（仅当它们是独立的后缀时）
        text = re.sub(r'》的$', '', text)
        text = re.sub(r'》精神$', '', text)
        text = re.sub(r'》要求$', '', text)
        text = re.sub(r'》规定$', '', text)
        
        return text.strip()
    
    def _normalize_citation(self, text: str) -> str:
        """
        标准化引用文本用于去重
        
        Args:
            text: 引用文本
        
        Returns:
            str: 标准化后的文本
        """
        if not text:
            return ""
        
        # 移除所有标点和空白
        normalized = re.sub(r'[《》「」「」\s]', '', text)
        return normalized.lower()
    
    def _is_valid_citation(self, cited_text: str) -> bool:
        """
        验证引用是否有效
        
        Args:
            cited_text: 被引用的文本
        
        Returns:
            bool: 是否为有效引用
        """
        if not cited_text:
            return False
        
        # 长度检查
        if len(cited_text) < 2 or len(cited_text) > 100:
            return False
        
        # 排除常见的非引用内容
        invalid_patterns = [
            r'^第[一二三四五六七八九十\d]+[条章节款项]',  # 条款引用
            r'^[一二三四五六七八九十\d]+[、.]',  # 序号
            r'^[\d]+$',  # 纯数字
            r'^[a-zA-Z\s]+$',  # 纯英文
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, cited_text):
                return False
        
        return True
    
    def get_all_citations(self) -> List[Citation]:
        """
        获取所有提取的引用
        
        Returns:
            List[Citation]: 引用列表
        """
        if not self.citations:
            self.extract_citations()
        return self.citations
    
    def get_citations_by_document(self, doc_name: str) -> List[Citation]:
        """
        获取指定文档的所有引用
        
        Args:
            doc_name: 文档名称
        
        Returns:
            List[Citation]: 该文档的引用列表
        """
        if not self.citations:
            self.extract_citations()
        return [c for c in self.citations if c.source_doc == doc_name]
    
    def get_cited_documents(self, doc_name: str) -> List[str]:
        """
        获取指定文档引用的所有文档
        
        Args:
            doc_name: 文档名称
        
        Returns:
            List[str]: 被引用的文档名列表
        """
        if self._citation_network is None:
            self.extract_citations()
        return self._citation_network.get(doc_name, [])
    
    def build_citation_network(self) -> Dict[str, Dict[str, List[str]]]:
        """
        构建引用关系网络
        
        Returns:
            Dict: 包含 'nodes' 和 'edges' 的网络数据
        
        Requirements: 6.3
        """
        if self._citation_network is None:
            self.extract_citations()
        
        # 收集所有节点（文档和被引用的文件）
        nodes: Set[str] = set(self.file_names)
        edges: List[Tuple[str, str]] = []
        
        for source_doc, cited_docs in self._citation_network.items():
            for cited_doc in cited_docs:
                nodes.add(cited_doc)
                edges.append((source_doc, cited_doc))
        
        return {
            'nodes': list(nodes),
            'edges': edges
        }
    
    def get_citation_count(self, doc_name: str) -> Tuple[int, int]:
        """
        获取文档的引用统计
        
        Args:
            doc_name: 文档名称
        
        Returns:
            Tuple[int, int]: (被引用次数, 引用其他文档次数)
        
        Requirements: 6.5
        """
        if self._citation_network is None:
            self.extract_citations()
        
        # 计算引用其他文档的次数
        cites_count = len(self._citation_network.get(doc_name, []))
        
        # 计算被引用次数
        cited_by_count = 0
        for source_doc, cited_docs in self._citation_network.items():
            if doc_name in cited_docs or any(doc_name in cd for cd in cited_docs):
                cited_by_count += 1
        
        return (cited_by_count, cites_count)
    
    def get_all_citation_stats(self) -> List[CitationStats]:
        """
        获取所有文档的引用统计
        
        Returns:
            List[CitationStats]: 引用统计列表
        """
        if self._citation_network is None:
            self.extract_citations()
        
        stats = []
        
        # 收集所有文档（包括被引用但不在文件列表中的）
        all_docs: Set[str] = set(self.file_names)
        for cited_docs in self._citation_network.values():
            all_docs.update(cited_docs)
        
        for doc_name in all_docs:
            # 计算被哪些文档引用
            cited_by = []
            for source_doc, cited_docs in self._citation_network.items():
                if doc_name in cited_docs:
                    cited_by.append(source_doc)
            
            # 计算引用了哪些文档
            cites = self._citation_network.get(doc_name, [])
            
            stats.append(CitationStats(
                doc_name=doc_name,
                cited_by_count=len(cited_by),
                cites_count=len(cites),
                cited_by=cited_by,
                cites=cites
            ))
        
        return stats
    
    def find_core_documents(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """
        识别引用网络中的核心文档（高被引文档）
        
        Args:
            top_n: 返回的核心文档数量
        
        Returns:
            List[Tuple[str, int]]: (文档名, 被引用次数) 列表，按被引用次数降序排列
        
        Requirements: 6.6
        """
        if self._citation_network is None:
            self.extract_citations()
        
        # 统计每个文档被引用的次数
        citation_counts: Counter = Counter()
        
        for source_doc, cited_docs in self._citation_network.items():
            for cited_doc in cited_docs:
                citation_counts[cited_doc] += 1
        
        # 按被引用次数降序排序
        sorted_docs = citation_counts.most_common(top_n)
        
        return sorted_docs
    
    def get_citation_matrix(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        生成引用矩阵
        
        Returns:
            Tuple[pd.DataFrame, List[str]]: (引用矩阵DataFrame, 文档名列表)
        """
        if self._citation_network is None:
            self.extract_citations()
        
        # 收集所有文档
        all_docs: Set[str] = set(self.file_names)
        for cited_docs in self._citation_network.values():
            all_docs.update(cited_docs)
        
        doc_list = sorted(list(all_docs))
        n = len(doc_list)
        
        # 创建矩阵
        matrix = np.zeros((n, n), dtype=int)
        doc_to_idx = {doc: i for i, doc in enumerate(doc_list)}
        
        for source_doc, cited_docs in self._citation_network.items():
            if source_doc in doc_to_idx:
                source_idx = doc_to_idx[source_doc]
                for cited_doc in cited_docs:
                    if cited_doc in doc_to_idx:
                        cited_idx = doc_to_idx[cited_doc]
                        matrix[source_idx][cited_idx] = 1
        
        df = pd.DataFrame(matrix, index=doc_list, columns=doc_list)
        return df, doc_list

    def export_network_data(self) -> str:
        """
        导出引用网络数据为CSV格式
        
        Returns:
            str: CSV格式字符串
        
        Requirements: 6.7
        """
        if self._citation_network is None:
            self.extract_citations()
        
        # 构建边列表数据
        data = []
        for source_doc, cited_docs in self._citation_network.items():
            for cited_doc in cited_docs:
                data.append({
                    "引用文档": source_doc,
                    "被引用文档": cited_doc
                })
        
        if not data:
            return ""
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='utf-8-sig')
    
    def export_citation_list(self) -> str:
        """
        导出引用列表为CSV格式
        
        Returns:
            str: CSV格式字符串
        """
        if not self.citations:
            self.extract_citations()
        
        data = []
        for citation in self.citations:
            data.append({
                "来源文档": citation.source_doc,
                "被引用内容": citation.cited_text,
                "引用上下文": citation.context,
                "位置": citation.position
            })
        
        if not data:
            return ""
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='utf-8-sig')
    
    def export_citation_stats(self) -> str:
        """
        导出引用统计为CSV格式
        
        Returns:
            str: CSV格式字符串
        """
        stats = self.get_all_citation_stats()
        
        data = []
        for stat in stats:
            data.append({
                "文档名": stat.doc_name,
                "被引用次数": stat.cited_by_count,
                "引用次数": stat.cites_count,
                "被引用来源": ", ".join(stat.cited_by[:5]),
                "引用目标": ", ".join(stat.cites[:5])
            })
        
        if not data:
            return ""
        
        df = pd.DataFrame(data)
        # 按被引用次数降序排序
        df = df.sort_values("被引用次数", ascending=False)
        return df.to_csv(index=False, encoding='utf-8-sig')
    
    def get_network_for_visualization(self) -> Tuple[List[Dict], List[Dict]]:
        """
        获取用于可视化的网络数据
        
        Returns:
            Tuple[List[Dict], List[Dict]]: (节点列表, 边列表)
        """
        if self._citation_network is None:
            self.extract_citations()
        
        # 统计被引用次数用于节点大小
        citation_counts: Counter = Counter()
        for cited_docs in self._citation_network.values():
            for cited_doc in cited_docs:
                citation_counts[cited_doc] += 1
        
        # 构建节点列表
        all_docs: Set[str] = set(self.file_names)
        for cited_docs in self._citation_network.values():
            all_docs.update(cited_docs)
        
        nodes = []
        for doc in all_docs:
            is_source = doc in self.file_names
            cited_count = citation_counts.get(doc, 0)
            nodes.append({
                "id": doc,
                "label": doc[:20] + "..." if len(doc) > 20 else doc,
                "size": 10 + cited_count * 5,
                "type": "source" if is_source else "cited",
                "cited_count": cited_count
            })
        
        # 构建边列表
        edges = []
        for source_doc, cited_docs in self._citation_network.items():
            for cited_doc in cited_docs:
                edges.append({
                    "source": source_doc,
                    "target": cited_doc
                })
        
        return nodes, edges


# ============================================================================
# Streamlit UI 渲染函数
# ============================================================================

def render_citation_analyzer():
    """
    渲染引用与参考分析模块UI
    
    Requirements: 6.4, 6.7
    """
    import streamlit as st
    from utils.session_state import log_message
    
    st.header("📖 引用与参考分析")
    
    # 功能介绍与操作手册
    with st.expander("📖 功能介绍与操作手册", expanded=False):
        st.markdown("""
        ## 📖 引用与参考分析模块
        
        **功能概述**：分析政策文本间的引用关系，了解政策的传承和影响脉络。
        
        ---
        
        ### 🎯 使用场景
        
        | 场景 | 关注点 | 应用 |
        |------|--------|------|
        | 政策溯源 | 引用关系 | 追踪政策的法律依据 |
        | 影响力分析 | 被引用次数 | 识别核心政策文件 |
        | 政策网络 | 引用网络 | 了解政策间的关联 |
        | 文献综述 | 引用列表 | 整理政策参考文献 |
        
        ---
        
        ### 📋 操作步骤
        
        **1. 提取引用**：
        - 点击"提取引用"按钮
        - 系统自动识别文本中的引用（如《xxx》、根据xxx规定等）
        
        **2. 查看引用列表**：
        - 查看每个文档的引用情况
        - 查看引用上下文
        
        **3. 引用网络分析**：
        - 查看引用关系有向图
        - 识别核心文档（高被引文档）
        
        **4. 导出结果**：
        - 导出引用列表
        - 导出引用网络数据
        - 导出引用统计
        
        ---
        
        ### ⚙️ 引用识别模式
        
        系统支持识别以下引用格式：
        - 《xxx》格式的文件引用
        - "根据xxx规定"格式
        - "依据xxx"格式
        - "按照xxx要求"格式
        - "参照xxx"格式
        - xxx号文格式
        - 《xxx通知/意见/办法》等
        
        ---
        
        ### 💡 使用建议
        
        **学术研究建议**：
        - 核心文档通常是被引用次数最多的文件
        - 引用网络可以揭示政策的层级关系
        - 结合时序分析可以追踪政策演变
        """)
    
    # 检查数据
    if not st.session_state.get("raw_texts"):
        st.warning("⚠️ 请先在「数据加载」标签页中加载文本数据")
        return
    
    raw_texts = st.session_state["raw_texts"]
    file_names = st.session_state.get("file_names", [])
    
    if not file_names:
        st.warning("⚠️ 未找到文档名称列表")
        return
    
    # 获取或创建分析器
    if "citation_analyzer" not in st.session_state or st.session_state["citation_analyzer"] is None:
        st.session_state["citation_analyzer"] = CitationAnalyzer(raw_texts, file_names)
    
    analyzer = st.session_state["citation_analyzer"]
    
    # 创建标签页
    tabs = st.tabs([
        "🔍 引用提取",
        "📋 引用列表",
        "🕸️ 引用网络",
        "⭐ 核心文档",
        "💾 导出"
    ])
    
    # ========== 引用提取 ==========
    with tabs[0]:
        st.subheader("引用提取")
        
        st.markdown("""
        点击下方按钮，系统将自动从文本中识别引用关系。
        
        **支持的引用格式**：
        - 《xxx》格式
        - 根据/依据/按照/参照xxx
        - xxx号文
        - 等等...
        """)
        
        if st.button("🔍 提取引用", type="primary", key="extract_citations"):
            with st.spinner("正在提取引用..."):
                citation_network = analyzer.extract_citations()
                st.session_state["citation_network"] = citation_network
                
                total_citations = len(analyzer.citations)
                docs_with_citations = len([d for d in citation_network.values() if d])
                
                log_message(f"提取了 {total_citations} 条引用，涉及 {docs_with_citations} 个文档")
                
                st.success(f"✅ 提取完成！共发现 {total_citations} 条引用")
        
        # 显示提取统计
        if analyzer.citations:
            st.markdown("---")
            st.subheader("📊 提取统计")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("总引用数", len(analyzer.citations))
            
            with col2:
                docs_with_citations = len([d for d in analyzer._citation_network.values() if d])
                st.metric("有引用的文档", docs_with_citations)
            
            with col3:
                unique_cited = set()
                for cited_docs in analyzer._citation_network.values():
                    unique_cited.update(cited_docs)
                st.metric("被引用文件数", len(unique_cited))
    
    # ========== 引用列表 ==========
    with tabs[1]:
        st.subheader("引用列表")
        
        if not analyzer.citations:
            st.info("💡 请先在「引用提取」标签页中提取引用")
            return
        
        # 按文档筛选
        selected_doc = st.selectbox(
            "选择文档查看引用",
            ["全部文档"] + file_names,
            key="citation_doc_filter"
        )
        
        if selected_doc == "全部文档":
            citations_to_show = analyzer.citations
        else:
            citations_to_show = analyzer.get_citations_by_document(selected_doc)
        
        if not citations_to_show:
            st.info("该文档没有引用其他文件")
        else:
            # 显示引用列表
            st.markdown(f"**共 {len(citations_to_show)} 条引用**")
            
            for i, citation in enumerate(citations_to_show[:50]):  # 最多显示50条
                with st.expander(f"📄 {citation.cited_text}", expanded=False):
                    st.markdown(f"**来源文档**: {citation.source_doc}")
                    st.markdown(f"**引用上下文**: ...{citation.context}...")
            
            if len(citations_to_show) > 50:
                st.info(f"仅显示前50条引用，共 {len(citations_to_show)} 条")
    
    # ========== 引用网络 ==========
    with tabs[2]:
        st.subheader("引用网络可视化")
        
        if not analyzer.citations:
            st.info("💡 请先在「引用提取」标签页中提取引用")
            return
        
        # 获取网络数据
        nodes, edges = analyzer.get_network_for_visualization()
        
        if not edges:
            st.info("未发现引用关系")
            return
        
        st.markdown(f"**网络规模**: {len(nodes)} 个节点, {len(edges)} 条边")
        
        # 尝试使用pyvis绘制网络图
        try:
            from pyvis.network import Network
            import tempfile
            import os
            
            # 创建网络
            net = Network(height="600px", width="100%", directed=True, 
                         bgcolor="#ffffff", font_color="#333333")
            
            # 添加节点
            for node in nodes:
                color = "#1E88E5" if node["type"] == "source" else "#FFA726"
                net.add_node(
                    node["id"],
                    label=node["label"],
                    size=node["size"],
                    color=color,
                    title=f"{node['id']}\n被引用: {node['cited_count']}次"
                )
            
            # 添加边
            for edge in edges:
                net.add_edge(edge["source"], edge["target"], arrows="to")
            
            # 设置物理布局
            net.set_options("""
            {
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 200,
                        "springConstant": 0.08
                    },
                    "maxVelocity": 50,
                    "solver": "forceAtlas2Based",
                    "timestep": 0.35,
                    "stabilization": {"iterations": 150}
                }
            }
            """)
            
            # 保存并显示
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
                net.save_graph(f.name)
                with open(f.name, 'r', encoding='utf-8') as html_file:
                    html_content = html_file.read()
                st.components.v1.html(html_content, height=620, scrolling=True)
                os.unlink(f.name)
            
            # 图例
            st.markdown("""
            **图例说明**：
            - 🔵 蓝色节点：源文档（您上传的文件）
            - 🟠 橙色节点：被引用文档
            - 箭头方向：从引用文档指向被引用文档
            - 节点大小：被引用次数越多，节点越大
            """)
            
        except ImportError:
            st.warning("💡 安装 pyvis 库可显示交互式网络图: `pip install pyvis`")
            
            # 使用简单的表格展示
            st.markdown("**引用关系表**")
            edge_df = pd.DataFrame(edges)
            edge_df.columns = ["引用文档", "被引用文档"]
            st.dataframe(edge_df, use_container_width=True, hide_index=True)
    
    # ========== 核心文档 ==========
    with tabs[3]:
        st.subheader("核心文档识别")
        
        if not analyzer.citations:
            st.info("💡 请先在「引用提取」标签页中提取引用")
            return
        
        # 获取核心文档
        top_n = st.slider("显示核心文档数量", 5, 20, 10, key="core_doc_count")
        core_docs = analyzer.find_core_documents(top_n=top_n)
        
        if not core_docs:
            st.info("未发现被引用的文档")
            return
        
        st.markdown("**核心文档（按被引用次数排序）**")
        
        # 显示核心文档列表
        core_df = pd.DataFrame(core_docs, columns=["文档名", "被引用次数"])
        
        # 添加排名
        core_df.insert(0, "排名", range(1, len(core_df) + 1))
        
        st.dataframe(core_df, use_container_width=True, hide_index=True)
        
        # 可视化
        if len(core_docs) > 0:
            st.markdown("---")
            st.markdown("**被引用次数分布**")
            
            try:
                import plotly.express as px
                
                fig = px.bar(
                    core_df,
                    x="文档名",
                    y="被引用次数",
                    title="核心文档被引用次数",
                    color="被引用次数",
                    color_continuous_scale="Blues"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
            except ImportError:
                # 使用Streamlit原生图表
                chart_df = core_df.set_index("文档名")["被引用次数"]
                st.bar_chart(chart_df)
        
        # 显示详细统计
        st.markdown("---")
        st.subheader("📊 详细引用统计")
        
        stats = analyzer.get_all_citation_stats()
        stats_data = []
        for stat in stats:
            stats_data.append({
                "文档名": stat.doc_name[:30] + "..." if len(stat.doc_name) > 30 else stat.doc_name,
                "被引用次数": stat.cited_by_count,
                "引用次数": stat.cites_count
            })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df = stats_df.sort_values("被引用次数", ascending=False)
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # ========== 导出 ==========
    with tabs[4]:
        st.subheader("导出引用分析结果")
        
        if not analyzer.citations:
            st.info("💡 请先在「引用提取」标签页中提取引用")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 引用列表**")
            citation_csv = analyzer.export_citation_list()
            if citation_csv:
                st.download_button(
                    label="📥 下载引用列表CSV",
                    data=citation_csv,
                    file_name="citation_list.csv",
                    mime="text/csv",
                    key="download_citation_list"
                )
            
            st.markdown("**🕸️ 引用网络**")
            network_csv = analyzer.export_network_data()
            if network_csv:
                st.download_button(
                    label="📥 下载引用网络CSV",
                    data=network_csv,
                    file_name="citation_network.csv",
                    mime="text/csv",
                    key="download_citation_network"
                )
        
        with col2:
            st.markdown("**📊 引用统计**")
            stats_csv = analyzer.export_citation_stats()
            if stats_csv:
                st.download_button(
                    label="📥 下载引用统计CSV",
                    data=stats_csv,
                    file_name="citation_stats.csv",
                    mime="text/csv",
                    key="download_citation_stats"
                )
            
            st.markdown("**📑 引用矩阵**")
            matrix_df, doc_list = analyzer.get_citation_matrix()
            if not matrix_df.empty:
                matrix_csv = matrix_df.to_csv(encoding='utf-8-sig')
                st.download_button(
                    label="📥 下载引用矩阵CSV",
                    data=matrix_csv,
                    file_name="citation_matrix.csv",
                    mime="text/csv",
                    key="download_citation_matrix"
                )
