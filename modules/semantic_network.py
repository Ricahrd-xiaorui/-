﻿﻿﻿﻿﻿﻿﻿﻿﻿# -*- coding: utf-8 -*-
"""
语义网络分析模块 (Semantic Network Analysis Module)

本模块提供语义网络构建和分析功能，包括：
- 基于词语共现构建语义网络
- 核心概念词过滤
- 社区检测
- 中心性指标计算
- 网络数据导出

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import pandas as pd

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class SemanticNetworkBuilder:
    """语义网络构建器"""
    
    def __init__(self, texts: List[List[str]], cooccurrence_data: Dict[Tuple[str, str], int]):
        self.texts = texts if texts else []
        self.cooccurrence_data = cooccurrence_data if cooccurrence_data else {}
        self.network = None
        self._community_labels = None
        self._centrality_metrics = None

    def build_network(self, min_weight: int = 2):
        """构建语义网络 - Requirements: 7.1"""
        if not HAS_NETWORKX:
            return None
        
        self.network = nx.Graph()
        
        for (word1, word2), freq in self.cooccurrence_data.items():
            if freq >= min_weight:
                self.network.add_edge(word1, word2, weight=freq)
        
        self._community_labels = None
        self._centrality_metrics = None
        
        return self.network
    
    def filter_by_center(self, center_word: str, max_depth: int = 2):
        """以指定词语为中心过滤网络 - Requirements: 7.2, 7.3"""
        if not HAS_NETWORKX or self.network is None:
            return None
        
        if center_word not in self.network:
            return None
        
        nodes_in_range = {center_word}
        current_level = {center_word}
        
        for _ in range(max_depth):
            next_level = set()
            for node in current_level:
                neighbors = set(self.network.neighbors(node))
                next_level.update(neighbors - nodes_in_range)
            nodes_in_range.update(next_level)
            current_level = next_level
            if not current_level:
                break
        
        subgraph = self.network.subgraph(nodes_in_range).copy()
        return subgraph

    def detect_communities(self) -> Dict[str, int]:
        """检测语义网络中的社区 - Requirements: 7.5"""
        if not HAS_NETWORKX or self.network is None:
            return {}
        
        if self.network.number_of_nodes() == 0:
            return {}
        
        if self._community_labels is not None:
            return self._community_labels
        
        try:
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(self.network)
                self._community_labels = partition
            except ImportError:
                from networkx.algorithms.community import greedy_modularity_communities
                communities = list(greedy_modularity_communities(self.network))
                
                self._community_labels = {}
                for community_id, community in enumerate(communities):
                    for node in community:
                        self._community_labels[node] = community_id
        except Exception:
            self._community_labels = {node: 0 for node in self.network.nodes()}
        
        return self._community_labels

    def calculate_centrality(self) -> Dict[str, Dict[str, float]]:
        """计算网络的中心性指标 - Requirements: 7.7"""
        if not HAS_NETWORKX or self.network is None:
            return {}
        
        if self.network.number_of_nodes() == 0:
            return {}
        
        if self._centrality_metrics is not None:
            return self._centrality_metrics
        
        self._centrality_metrics = {}
        
        degree_centrality = nx.degree_centrality(self.network)
        
        try:
            betweenness_centrality = nx.betweenness_centrality(self.network)
        except Exception:
            betweenness_centrality = {node: 0.0 for node in self.network.nodes()}
        
        try:
            closeness_centrality = nx.closeness_centrality(self.network)
        except Exception:
            closeness_centrality = {node: 0.0 for node in self.network.nodes()}
        
        try:
            eigenvector_centrality = nx.eigenvector_centrality(self.network, max_iter=1000)
        except Exception:
            eigenvector_centrality = {node: 0.0 for node in self.network.nodes()}
        
        for node in self.network.nodes():
            self._centrality_metrics[node] = {
                'degree': degree_centrality.get(node, 0.0),
                'betweenness': betweenness_centrality.get(node, 0.0),
                'closeness': closeness_centrality.get(node, 0.0),
                'eigenvector': eigenvector_centrality.get(node, 0.0)
            }
        
        return self._centrality_metrics

    def get_top_central_nodes(self, metric: str = 'degree', top_n: int = 10) -> List[Tuple[str, float]]:
        """获取中心性最高的节点"""
        centrality = self.calculate_centrality()
        if not centrality:
            return []
        
        metric_values = [(node, metrics.get(metric, 0.0)) for node, metrics in centrality.items()]
        sorted_nodes = sorted(metric_values, key=lambda x: -x[1])
        return sorted_nodes[:top_n]
    
    def get_network_statistics(self) -> Dict[str, any]:
        """获取网络统计信息"""
        if not HAS_NETWORKX or self.network is None:
            return {}
        
        stats = {
            'num_nodes': self.network.number_of_nodes(),
            'num_edges': self.network.number_of_edges(),
            'density': nx.density(self.network) if self.network.number_of_nodes() > 0 else 0,
            'avg_degree': sum(dict(self.network.degree()).values()) / max(self.network.number_of_nodes(), 1),
        }
        
        if self.network.number_of_nodes() > 0:
            stats['num_components'] = nx.number_connected_components(self.network)
            if stats['num_components'] > 0:
                largest_cc = max(nx.connected_components(self.network), key=len)
                stats['largest_component_size'] = len(largest_cc)
            else:
                stats['largest_component_size'] = 0
        else:
            stats['num_components'] = 0
            stats['largest_component_size'] = 0
        
        # 使用缓存的社区数据，不自动计算
        if self._community_labels:
            stats['num_communities'] = len(set(self._community_labels.values()))
        else:
            stats['num_communities'] = 0
        
        return stats

    def export_network(self) -> Tuple[str, str]:
        """导出语义网络数据 - Requirements: 7.8"""
        if not HAS_NETWORKX or self.network is None:
            return "", ""
        
        communities = self.detect_communities()
        centrality = self.calculate_centrality()
        
        nodes_data = []
        for node in self.network.nodes():
            node_data = {
                '节点': node,
                '度数': self.network.degree(node),
                '社区': communities.get(node, 0),
            }
            if node in centrality:
                node_data['度中心性'] = round(centrality[node].get('degree', 0), 4)
                node_data['介数中心性'] = round(centrality[node].get('betweenness', 0), 4)
                node_data['接近中心性'] = round(centrality[node].get('closeness', 0), 4)
            nodes_data.append(node_data)
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_csv = nodes_df.to_csv(index=False, encoding='utf-8-sig')
        
        edges_data = []
        for u, v, data in self.network.edges(data=True):
            edges_data.append({
                '源节点': u,
                '目标节点': v,
                '权重': data.get('weight', 1)
            })
        
        edges_df = pd.DataFrame(edges_data)
        edges_csv = edges_df.to_csv(index=False, encoding='utf-8-sig')
        
        return nodes_csv, edges_csv

    def to_vis_data(self, max_nodes: int = 100) -> Tuple[List[dict], List[dict]]:
        """转换为可视化数据格式"""
        if not HAS_NETWORKX or self.network is None:
            return [], []
        
        # 不自动计算社区和中心性，使用缓存的数据
        communities = self._community_labels if self._community_labels else {}
        centrality = self._centrality_metrics if self._centrality_metrics else {}
        
        node_degrees = dict(self.network.degree())
        sorted_nodes = sorted(node_degrees.items(), key=lambda x: -x[1])[:max_nodes]
        top_nodes = set(node for node, _ in sorted_nodes)
        
        nodes = []
        for node, degree in sorted_nodes:
            node_data = {
                'id': node,
                'label': node,
                'size': degree,
                'community': communities.get(node, 0),
            }
            if node in centrality:
                node_data['degree_centrality'] = centrality[node].get('degree', 0)
                node_data['betweenness'] = centrality[node].get('betweenness', 0)
            nodes.append(node_data)
        
        edges = []
        for u, v, data in self.network.edges(data=True):
            if u in top_nodes and v in top_nodes:
                edges.append({
                    'source': u,
                    'target': v,
                    'weight': data.get('weight', 1)
                })
        
        return nodes, edges


# ============================================================================
# Streamlit UI 渲染函数
# ============================================================================

def render_semantic_network():
    """渲染语义网络分析模块UI - Requirements: 7.4, 7.6, 7.8"""
    import streamlit as st
    from utils.session_state import log_message
    
    st.header("🕸️ 语义网络分析")
    
    # 功能介绍与操作手册
    with st.expander("📖 功能介绍与操作手册", expanded=False):
        st.markdown("""
        ## 🕸️ 语义网络分析模块
        
        **功能概述**：基于词语共现关系构建语义网络，通过社区检测和中心性分析揭示文本中的概念结构和关键词语。
        
        ---
        
        ### 🎯 核心功能
        
        | 功能 | 说明 | 应用场景 |
        |------|------|----------|
        | 网络构建 | 基于共现关系构建词语网络 | 概念关系可视化 |
        | 社区检测 | 识别语义相近的词语群组 | 主题聚类、概念分组 |
        | 中心性分析 | 计算节点重要性指标 | 识别核心概念 |
        | 子网络提取 | 以核心词为中心提取子网络 | 聚焦特定概念 |
        
        ---
        
        ### 📊 中心性指标说明
        
        | 指标 | 含义 | 高值表示 |
        |------|------|----------|
        | 度中心性 | 节点的连接数量 | 与多个词语共现，活跃度高 |
        | 介数中心性 | 节点作为桥梁的程度 | 连接不同概念群的关键词 |
        | 接近中心性 | 到其他节点的平均距离 | 处于网络中心位置 |
        | 特征向量中心性 | 连接重要节点的程度 | 与重要词语关联密切 |
        
        ---
        
        ### 📋 操作步骤
        
        **1. 准备数据**
        - 确保已完成文本预处理
        - 在「基础文本分析」中计算词语共现关系
        - 或使用本页面的「快速计算」功能
        
        **2. 设置参数**
        - **最小边权重**：过滤低频共现关系（建议2-5）
        - **最大节点数**：控制可视化复杂度（建议30-100）
        - **核心概念词**：可选，聚焦特定概念的子网络
        
        **3. 构建网络**
        - 点击「构建语义网络」按钮
        - 查看网络可视化和统计信息
        
        **4. 分析结果**
        - 观察社区分布（不同颜色代表不同社区）
        - 查看中心性排名，识别核心概念
        - 导出网络数据用于进一步分析
        
        ---
        
        ### 💡 使用建议
        
        - **参数调优**：从较高的最小边权重开始，逐步降低以观察更多关系
        - **核心概念**：输入研究关注的关键词，可获得更聚焦的子网络
        - **社区解读**：同一社区的词语通常语义相关，可作为主题标签
        - **学术应用**：中心性指标可用于识别文本中的核心概念和关键议题
        
        ---
        
        ### 📁 导出数据
        
        - **节点列表**：包含词语、度数、社区、中心性等信息
        - **边列表**：包含词语对和共现权重
        - 可用于 Gephi、Pajek 等专业网络分析软件
        """)
    
    if not st.session_state.get("texts"):
        st.warning("请先在「文本预处理」标签页中完成文本预处理")
        return
    
    cooccurrence_data = st.session_state.get("cooccurrence_matrix", {})
    if not cooccurrence_data:
        st.info("💡 请先在「基础文本分析」「词语共现分析」中计算共现关系")
        
        st.markdown("---")
        st.subheader("快速计算共现关系")
        
        col1, col2 = st.columns(2)
        with col1:
            quick_window_size = st.slider("共现窗口大小", min_value=2, max_value=20, value=5, key="semantic_quick_window")
        with col2:
            quick_min_freq = st.slider("最小共现频率", min_value=1, max_value=20, value=2, key="semantic_quick_min_freq")
        
        if st.button("计算共现关系", key="semantic_calc_cooc"):
            from modules.word_frequency_cooccurrence import CooccurrenceAnalyzer
            
            texts = st.session_state["texts"]
            
            if not texts or all(len(text) == 0 for text in texts):
                st.error("❌ 文本数据为空，请先完成文本预处理")
                return
            
            # 创建进度条
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            try:
                analyzer = CooccurrenceAnalyzer(texts, quick_window_size)
                
                # 定义进度回调函数
                def update_progress(current, total):
                    progress = current / total if total > 0 else 0
                    progress_bar.progress(progress)
                    progress_text.text(f"正在计算共现关系... {current}/{total} 个文档 ({progress*100:.1f}%)")
                
                progress_text.text("正在初始化...")
                
                # 先计算所有共现关系（带进度显示）
                all_cooccurrence = analyzer.calculate_cooccurrence(progress_callback=update_progress)
                
                progress_text.text("正在过滤结果...")
                
                if not all_cooccurrence:
                    progress_bar.empty()
                    progress_text.empty()
                    st.warning("⚠️ 未找到任何共现词语对，可能原因：\n- 文本太短\n- 窗口大小设置不当\n- 文本中词语重复度低")
                    return
                
                # 再进行过滤
                cooccurrence = analyzer.filter_by_threshold(quick_min_freq)
                
                progress_bar.empty()
                progress_text.empty()
                
                if cooccurrence:
                    st.session_state["cooccurrence_matrix"] = cooccurrence
                    st.session_state["cooccurrence_analyzer"] = analyzer
                    st.success(f"✅ 共现分析完成！找到 {len(cooccurrence)} 对共现词语（总共 {len(all_cooccurrence)} 对，过滤后保留 {len(cooccurrence)} 对）")
                    log_message(f"快速计算共现关系完成，窗口={quick_window_size}，最小频率={quick_min_freq}，结果={len(cooccurrence)}对")
                    st.rerun()
                else:
                    st.warning(f"⚠️ 过滤后未找到符合条件的共现词语对\n\n"
                             f"- 总共找到 {len(all_cooccurrence)} 对共现词语\n"
                             f"- 但没有词对的共现频率 ≥ {quick_min_freq}\n"
                             f"- 建议：降低「最小共现频率」参数")
                    
                    # 显示频率分布
                    freq_dist = {}
                    for freq in all_cooccurrence.values():
                        freq_dist[freq] = freq_dist.get(freq, 0) + 1
                    
                    st.info("📊 共现频率分布：")
                    for freq in sorted(freq_dist.keys(), reverse=True):
                        st.write(f"  - 频率 {freq}: {freq_dist[freq]} 对词语")
            
            except Exception as e:
                progress_bar.empty()
                progress_text.empty()
                st.error(f"❌ 计算共现关系时出错：{type(e).__name__}: {str(e)}")
                import traceback
                with st.expander("查看详细错误信息"):
                    st.code(traceback.format_exc())
        return
    
    texts = st.session_state["texts"]
    
    st.subheader("⚙️ 网络构建参数")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_weight = st.slider("最小边权重", min_value=1, max_value=20, value=3, help="过滤共现频率低于此值的边（建议3-5）", key="semantic_min_weight")
    
    with col2:
        max_nodes = st.slider("最大节点数", min_value=20, max_value=200, value=30, help="可视化显示的最大节点数量（建议30-50）", key="semantic_max_nodes")
    
    with col3:
        center_word = st.text_input("核心概念词（可选）", value=st.session_state.get("center_word", ""), help="输入核心概念词，将显示以该词为中心的子网络", key="semantic_center_word")
        st.session_state["center_word"] = center_word
    
    if center_word:
        max_depth = st.slider("网络深度", min_value=1, max_value=3, value=2, help="从核心概念扩展的最大跳数", key="semantic_max_depth")
    else:
        max_depth = 2

    if st.button("🔨 构建语义网络", type="primary", key="semantic_build_btn"):
        if not HAS_NETWORKX:
            st.error("需要安装networkx库: pip install networkx")
            return
        
        # 清除之前的强制可视化确认状态
        if "force_viz_confirmed" in st.session_state:
            del st.session_state["force_viz_confirmed"]
        
        # 添加进度显示
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        try:
            progress_text.text("正在构建网络图...")
            progress_bar.progress(0.2)
            
            builder = SemanticNetworkBuilder(texts, cooccurrence_data)
            network = builder.build_network(min_weight=min_weight)
            
            if network and network.number_of_nodes() > 0:
                progress_bar.progress(0.4)
                
                # 检查网络大小
                num_nodes = network.number_of_nodes()
                num_edges = network.number_of_edges()
                
                if num_nodes > 100:
                    st.warning(f"⚠️ 网络较大（{num_nodes} 个节点），社区检测和中心性计算可能需要较长时间")
                
                progress_text.text("正在过滤网络...")
                
                if center_word and center_word in network:
                    filtered_network = builder.filter_by_center(center_word, max_depth)
                    if filtered_network:
                        builder.network = filtered_network
                        st.info(f"已过滤为以「{center_word}」为中心的子网络")
                        num_nodes = builder.network.number_of_nodes()
                        num_edges = builder.network.number_of_edges()
                elif center_word:
                    st.warning(f"核心概念词「{center_word}」不在网络中，显示完整网络")
                
                progress_bar.progress(0.6)
                
                # 保存网络（不立即计算社区和中心性）
                st.session_state["semantic_network"] = builder.network
                st.session_state["semantic_network_builder"] = builder
                
                # 只有在小网络时才自动计算
                if num_nodes <= 50:
                    progress_text.text("正在检测社区...")
                    progress_bar.progress(0.7)
                    st.session_state["community_labels"] = builder.detect_communities()
                    
                    progress_text.text("正在计算中心性...")
                    progress_bar.progress(0.9)
                    st.session_state["centrality_metrics"] = builder.calculate_centrality()
                else:
                    # 大网络延迟计算
                    st.session_state["community_labels"] = {}
                    st.session_state["centrality_metrics"] = {}
                    st.info("💡 网络较大，社区检测和中心性分析将在需要时计算")
                
                progress_bar.progress(1.0)
                progress_text.text("完成！")
                
                log_message(f"语义网络构建完成，{num_nodes}个节点，{num_edges}条边")
                st.success(f"✅ 语义网络构建完成！{num_nodes} 个节点，{num_edges} 条边")
                
                # 清理进度显示
                import time
                time.sleep(0.5)
                progress_bar.empty()
                progress_text.empty()
            else:
                progress_bar.empty()
                progress_text.empty()
                st.warning("无法构建语义网络，请检查共现数据或降低最小边权重")
        
        except Exception as e:
            progress_bar.empty()
            progress_text.empty()
            st.error(f"❌ 构建语义网络时出错：{type(e).__name__}: {str(e)}")
            import traceback
            with st.expander("查看详细错误信息"):
                st.code(traceback.format_exc())
    
    if st.session_state.get("semantic_network") is not None:
        builder = st.session_state.get("semantic_network_builder")
        network = st.session_state["semantic_network"]
        communities = st.session_state.get("community_labels", {})
        centrality = st.session_state.get("centrality_metrics", {})
        
        if builder is None:
            builder = SemanticNetworkBuilder(texts, cooccurrence_data)
            builder.network = network
            builder._community_labels = communities
            builder._centrality_metrics = centrality
            st.session_state["semantic_network_builder"] = builder
        
        stats = builder.get_network_statistics()
        
        st.markdown("---")
        st.subheader("📊 网络统计")
        stat_cols = st.columns(6)
        stat_cols[0].metric("节点数", stats.get('num_nodes', 0))
        stat_cols[1].metric("边数", stats.get('num_edges', 0))
        stat_cols[2].metric("网络密度", f"{stats.get('density', 0):.4f}")
        stat_cols[3].metric("平均度", f"{stats.get('avg_degree', 0):.2f}")
        stat_cols[4].metric("连通分量", stats.get('num_components', 0))
        stat_cols[5].metric("社区数", stats.get('num_communities', 0))
        
        st.markdown("---")
        
        result_tabs = st.tabs(["🕸️ 网络可视化", "👥 社区分析", "📈 中心性分析", "💾 数据导出"])
        
        with result_tabs[0]:
            _render_network_visualization(builder)
        
        with result_tabs[1]:
            _render_community_analysis(builder, communities)
        
        with result_tabs[2]:
            _render_centrality_analysis(builder, centrality)
        
        with result_tabs[3]:
            _render_export_section(builder)


def _render_network_visualization(builder):
    """渲染网络可视化"""
    import streamlit as st
    
    st.subheader("语义网络图")
    
    # 检查网络大小
    num_nodes = builder.network.number_of_nodes() if builder.network else 0
    num_edges = builder.network.number_of_edges() if builder.network else 0
    
    if num_nodes == 0:
        st.warning("⚠️ 网络为空，无法可视化")
        return
    
    # 显示网络信息
    info_col1, info_col2 = st.columns([2, 1])
    with info_col1:
        st.info(f"📊 当前网络：{num_nodes} 个节点，{num_edges} 条边")
    
    # 检查是否需要强制可视化确认
    need_force_confirmation = num_nodes > 500
    force_viz_confirmed = st.session_state.get("force_viz_confirmed", False)
    
    # 网络大小警告和建议
    if need_force_confirmation and not force_viz_confirmed:
        # 超大网络，需要用户确认
        st.error(f"⚠️ 网络非常大（{num_nodes} 个节点），强烈建议优化：\n\n"
                 f"1. 返回上方，提高「最小边权重」到5或更高\n"
                 f"2. 使用「核心概念词」过滤功能\n"
                 f"3. 重新构建更小的网络\n\n"
                 f"如果坚持可视化，请：\n"
                 f"- 减少「最大显示节点数」到50以下\n"
                 f"- 使用「circular」或「random」布局（最快）\n"
                 f"- 关闭节点标签显示\n"
                 f"- 预计生成时间：1-3分钟")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("✅ 我了解风险，继续", type="primary", use_container_width=True):
                st.session_state["force_viz_confirmed"] = True
                st.rerun()
        with col2:
            if st.button("🔙 返回优化参数", use_container_width=True):
                st.info("💡 请返回上方调整网络构建参数")
        
        st.info("💡 点击上方按钮以继续")
        return
    else:
        # 显示适当的提示信息
        if num_nodes > 500:
            st.success(f"✅ 已确认可视化大网络（{num_nodes} 个节点）")
            st.warning("⚠️ 建议：减少节点数、使用circular布局、关闭标签")
        elif num_nodes > 300:
            st.warning(f"⚠️ 网络较大（{num_nodes} 个节点），建议：\n"
                       f"- 减少「最大显示节点数」到50以下\n"
                       f"- 使用「circular」或「random」布局（更快）\n"
                       f"- 关闭节点标签显示\n"
                       f"- 预计生成时间：30秒-1分钟")
        elif num_nodes > 150:
            st.info(f"💡 网络中等规模（{num_nodes} 个节点），建议：\n"
                    f"- 使用「circular」或「spring」布局\n"
                    f"- 预计生成时间：10-30秒")
        elif num_nodes > 100:
            st.info(f"💡 网络适中（{num_nodes} 个节点），可以正常使用所有布局算法")
    
    st.markdown("---")
    
    # 可视化参数设置
    st.markdown("### 🎨 可视化参数")
    
    # 性能模式选择（对于大网络）
    if num_nodes > 100:
        perf_mode = st.radio(
            "性能模式",
            ["快速模式（推荐）", "标准模式", "完整模式"],
            index=0 if num_nodes > 200 else 1,
            horizontal=True,
            help="快速模式：更少节点+简单布局；标准模式：平衡；完整模式：最佳效果但较慢"
        )
        
        # 根据性能模式调整默认值
        if perf_mode == "快速模式（推荐）":
            default_max_nodes = min(30, num_nodes)
            default_layout_idx = 1  # circular
            default_show_labels = False
        elif perf_mode == "标准模式":
            default_max_nodes = min(50, num_nodes)
            default_layout_idx = 1 if num_nodes > 150 else 0
            default_show_labels = num_nodes <= 100
        else:  # 完整模式
            default_max_nodes = min(100, num_nodes)
            default_layout_idx = 0  # spring
            default_show_labels = num_nodes <= 50
    else:
        default_max_nodes = min(50, num_nodes)
        default_layout_idx = 0
        default_show_labels = True
    
    param_col1, param_col2, param_col3, param_col4 = st.columns(4)
    
    with param_col1:
        # 根据网络大小和性能模式动态调整
        if num_nodes > 500:
            max_limit = 100
        elif num_nodes > 300:
            max_limit = 150
        elif num_nodes > 100:
            max_limit = 200
        else:
            max_limit = min(300, num_nodes)
        
        max_nodes = st.number_input(
            "最大显示节点数",
            min_value=10,
            max_value=max_limit,
            value=default_max_nodes,
            step=10,
            help=f"显示度数最高的前N个节点（当前网络共{num_nodes}个节点）",
            key="viz_max_nodes"
        )
    
    with param_col2:
        layout_algorithm = st.selectbox(
            "布局算法", 
            ["spring (力导向)", "circular (圆形)", "shell (同心圆)", "random (随机)"], 
            index=default_layout_idx,
            key="semantic_layout_algo",
            help="力导向效果最好但较慢，圆形和随机最快"
        )
    
    with param_col3:
        color_by = st.selectbox(
            "节点颜色",
            ["度数", "社区", "度中心性", "介数中心性"],
            index=0,
            key="semantic_color_by"
        )
    
    with param_col4:
        show_labels = st.checkbox(
            "显示标签",
            value=default_show_labels,
            key="semantic_show_labels",
            help="节点过多时建议关闭"
        )
    
    # 生成按钮
    generate_col1, generate_col2 = st.columns([3, 1])
    with generate_col1:
        if st.button("🎨 生成网络图", key="generate_semantic_viz", type="primary", use_container_width=True):
            # 预估生成时间
            if num_nodes > 300:
                estimated_time = "1-3分钟"
            elif num_nodes > 150:
                estimated_time = "30秒-1分钟"
            elif num_nodes > 100:
                estimated_time = "10-30秒"
            else:
                estimated_time = "5-10秒"
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text(f"正在生成网络图... 预计需要 {estimated_time}")
            
            try:
                # 步骤1: 获取可视化数据
                progress_bar.progress(0.1)
                status_text.text("步骤 1/5: 准备节点数据...")
                
                nodes, edges = builder.to_vis_data(max_nodes)
                
                if not nodes:
                    progress_bar.empty()
                    status_text.empty()
                    st.warning("⚠️ 没有可显示的节点，请检查网络数据")
                    return
                
                # 步骤2: 计算布局
                progress_bar.progress(0.3)
                status_text.text(f"步骤 2/5: 计算网络布局（{len(nodes)}个节点）...")
                
                import plotly.graph_objects as go
                
                layout_name = layout_algorithm.split()[0]
                pos = _get_network_layout(builder.network, layout_name, max_nodes)
                
                if not pos:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ 无法计算网络布局")
                    return
                
                # 步骤3: 准备着色数据
                progress_bar.progress(0.5)
                status_text.text("步骤 3/5: 准备节点着色...")
                
                # 只在需要时获取社区和中心性
                communities = {}
                centrality = {}
                
                if color_by == "社区":
                    communities = st.session_state.get("community_labels", {})
                    if not communities:
                        st.info("💡 需要先在「社区分析」标签页进行社区检测")
                        color_by = "度数"
                elif color_by in ["度中心性", "介数中心性"]:
                    centrality = st.session_state.get("centrality_metrics", {})
                    if not centrality:
                        st.info("💡 需要先在「中心性分析」标签页计算中心性指标")
                        color_by = "度数"
                
                # 步骤4: 构建节点和边数据
                progress_bar.progress(0.7)
                status_text.text("步骤 4/5: 构建图表数据...")
                
                node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
                
                for node_data in nodes:
                    node_id = node_data['id']
                    if node_id in pos:
                        node_x.append(pos[node_id][0])
                        node_y.append(pos[node_id][1])
                        
                        degree = node_data.get('size', 1)
                        
                        # 构建悬停文本
                        hover_parts = [f"<b>{node_id}</b>", f"度数: {degree}"]
                        if communities and node_id in communities:
                            hover_parts.append(f"社区: {communities[node_id]}")
                        node_text.append("<br>".join(hover_parts))
                        
                        # 确定颜色
                        if color_by == "社区" and communities:
                            node_color.append(communities.get(node_id, 0))
                        elif color_by == "度中心性" and centrality:
                            node_color.append(centrality.get(node_id, {}).get('degree', 0))
                        elif color_by == "介数中心性" and centrality:
                            node_color.append(centrality.get(node_id, {}).get('betweenness', 0))
                        else:
                            node_color.append(degree)
                        
                        node_size.append(max(10, min(50, degree * 3)))
                
                edge_x, edge_y = [], []
                
                for edge in edges:
                    source, target = edge['source'], edge['target']
                    if source in pos and target in pos:
                        edge_x.extend([pos[source][0], pos[target][0], None])
                        edge_y.extend([pos[source][1], pos[target][1], None])
                
                # 步骤5: 渲染图表
                progress_bar.progress(0.9)
                status_text.text("步骤 5/5: 渲染图表...")
                
                fig = go.Figure()
                
                # 添加边
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    showlegend=False,
                    name='边'
                ))
                
                # 添加节点
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text' if show_labels else 'markers',
                    marker=dict(
                        size=node_size,
                        color=node_color,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=color_by, thickness=15)
                    ),
                    text=[n['label'] for n in nodes if n['id'] in pos] if show_labels else None,
                    textposition='top center',
                    textfont=dict(size=9),
                    hovertext=node_text,
                    hoverinfo='text',
                    showlegend=False,
                    name='节点'
                ))
                
                fig.update_layout(
                    title=dict(
                        text=f"语义网络图 ({len(nodes)}个节点, {len(edges)}条边)",
                        x=0.5,
                        xanchor='center'
                    ),
                    showlegend=False,
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=700,
                    plot_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 完成
                progress_bar.progress(1.0)
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"✅ 网络图生成完成！显示了 {len(nodes)} 个节点，{len(edges)} 条边")
                
            except ImportError as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ 需要安装plotly库: pip install plotly")
                st.markdown("**网络节点列表（按度数排序）：**")
                try:
                    nodes, edges = builder.to_vis_data(max_nodes)
                    node_df = pd.DataFrame(nodes)
                    st.dataframe(node_df, use_container_width=True)
                except:
                    pass
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ 生成网络图时出错：{type(e).__name__}: {str(e)}")
                import traceback
                with st.expander("查看详细错误信息"):
                    st.code(traceback.format_exc())
    
    with generate_col2:
        st.write("")  # 占位
    
    # 提示信息
    if not st.session_state.get("semantic_viz_generated"):
        st.info("👆 设置好参数后，点击上方按钮生成网络图")


def _get_network_layout(network, layout_name: str, max_nodes: int) -> Dict[str, Tuple[float, float]]:
    """获取网络布局 - 优化版本"""
    if not HAS_NETWORKX or network is None:
        return {}
    
    try:
        # 对于大网络，只取前max_nodes个节点
        if network.number_of_nodes() > max_nodes:
            node_degrees = dict(network.degree())
            top_nodes = sorted(node_degrees.items(), key=lambda x: -x[1])[:max_nodes]
            top_node_ids = [n for n, _ in top_nodes]
            subgraph = network.subgraph(top_node_ids)
        else:
            subgraph = network
        
        # 根据网络大小调整迭代次数
        num_nodes = subgraph.number_of_nodes()
        if num_nodes > 100:
            iterations = 20  # 大网络用更少迭代
        elif num_nodes > 50:
            iterations = 30
        else:
            iterations = 50
        
        # 计算布局
        if layout_name == "spring":
            pos = nx.spring_layout(subgraph, k=2, iterations=iterations)
        elif layout_name == "circular":
            pos = nx.circular_layout(subgraph)
        elif layout_name == "shell":
            pos = nx.shell_layout(subgraph)
        elif layout_name == "random":
            pos = nx.random_layout(subgraph)
        else:
            # 默认使用spring布局
            pos = nx.spring_layout(subgraph, k=2, iterations=iterations)
        
        return pos
    except Exception:
        return {}


def _render_community_analysis(builder, communities: Dict[str, int]):
    """渲染社区分析"""
    import streamlit as st
    
    st.subheader("社区检测结果")
    
    # 检查是否已计算社区
    if not communities:
        num_nodes = builder.network.number_of_nodes() if builder.network else 0
        
        if num_nodes > 200:
            st.warning(f"⚠️ 网络较大（{num_nodes} 个节点），社区检测可能需要1-2分钟")
        
        if st.button("🔬 开始社区检测", key="detect_communities_btn"):
            with st.spinner("正在检测社区..."):
                try:
                    communities = builder.detect_communities()
                    st.session_state["community_labels"] = communities
                    # 不使用st.rerun()，直接显示结果
                    st.success(f"✅ 社区检测完成！共检测到 {len(set(communities.values()))} 个社区")
                    
                    # 直接显示结果，不刷新页面
                    community_members = {}
                    for node, community_id in communities.items():
                        if community_id not in community_members:
                            community_members[community_id] = []
                        community_members[community_id].append(node)
                    
                    sorted_communities = sorted(community_members.items(), key=lambda x: -len(x[1]))
                    
                    st.markdown(f"**共检测到 {len(sorted_communities)} 个社区**")
                    
                    for community_id, members in sorted_communities:
                        with st.expander(f"社区 {community_id + 1} ({len(members)} 个成员)", expanded=community_id < 3):
                            if builder.network:
                                member_degrees = [(m, builder.network.degree(m)) for m in members]
                                sorted_members = sorted(member_degrees, key=lambda x: -x[1])
                                member_df = pd.DataFrame(sorted_members, columns=['词语', '度数'])
                                st.dataframe(member_df, width='stretch', hide_index=True)
                            else:
                                st.write(", ".join(members[:20]))
                                if len(members) > 20:
                                    st.write(f"... 等共 {len(members)} 个词语")
                    
                except Exception as e:
                    st.error(f"❌ 社区检测失败：{str(e)}")
                    import traceback
                    with st.expander("查看详细错误信息"):
                        st.code(traceback.format_exc())
                    return
        else:
            st.info("👆 点击按钮开始社区检测")
            return
    else:
        # 已有社区数据，直接显示
        community_members = {}
        for node, community_id in communities.items():
            if community_id not in community_members:
                community_members[community_id] = []
            community_members[community_id].append(node)
        
        sorted_communities = sorted(community_members.items(), key=lambda x: -len(x[1]))
        
        st.markdown(f"**共检测到 {len(sorted_communities)} 个社区**")
        
        for community_id, members in sorted_communities:
            with st.expander(f"社区 {community_id + 1} ({len(members)} 个成员)", expanded=community_id < 3):
                if builder.network:
                    member_degrees = [(m, builder.network.degree(m)) for m in members]
                    sorted_members = sorted(member_degrees, key=lambda x: -x[1])
                    member_df = pd.DataFrame(sorted_members, columns=['词语', '度数'])
                    st.dataframe(member_df, width='stretch', hide_index=True)
                else:
                    st.write(", ".join(members[:20]))
                    if len(members) > 20:
                        st.write(f"... 等共 {len(members)} 个词语")


def _render_centrality_analysis(builder, centrality: Dict[str, Dict[str, float]]):
    """渲染中心性分析"""
    import streamlit as st
    
    st.subheader("中心性指标分析")
    
    # 检查是否已计算中心性
    if not centrality:
        num_nodes = builder.network.number_of_nodes() if builder.network else 0
        
        if num_nodes > 200:
            st.warning(f"⚠️ 网络较大（{num_nodes} 个节点），中心性计算可能需要1-2分钟")
        
        if st.button("📊 开始中心性计算", key="calc_centrality_btn"):
            with st.spinner("正在计算中心性指标..."):
                try:
                    centrality = builder.calculate_centrality()
                    st.session_state["centrality_metrics"] = centrality
                    # 不使用st.rerun()，直接显示结果
                    st.success("✅ 中心性计算完成！")
                    
                    # 直接显示结果
                    _display_centrality_results(builder, centrality)
                    
                except Exception as e:
                    st.error(f"❌ 中心性计算失败：{str(e)}")
                    import traceback
                    with st.expander("查看详细错误信息"):
                        st.code(traceback.format_exc())
                    return
        else:
            st.info("👆 点击按钮开始计算中心性指标")
            return
    else:
        # 已有中心性数据，直接显示
        _display_centrality_results(builder, centrality)


def _display_centrality_results(builder, centrality: Dict[str, Dict[str, float]]):
    """显示中心性分析结果（提取为独立函数避免重复代码）"""
    import streamlit as st
    
    metric_names = {
        'degree': '度中心性',
        'betweenness': '介数中心性',
        'closeness': '接近中心性',
        'eigenvector': '特征向量中心性'
    }
    
    selected_metric = st.selectbox("选择中心性指标", list(metric_names.keys()), format_func=lambda x: metric_names[x], key="semantic_centrality_metric")
    
    top_n = st.slider("显示前N个节点", min_value=5, max_value=50, value=20, key="semantic_centrality_top_n")
    top_nodes = builder.get_top_central_nodes(selected_metric, top_n)
    
    if top_nodes:
        df = pd.DataFrame(top_nodes, columns=['词语', metric_names[selected_metric]])
        df['排名'] = range(1, len(df) + 1)
        df = df[['排名', '词语', metric_names[selected_metric]]]
        df[metric_names[selected_metric]] = df[metric_names[selected_metric]].round(4)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(df, width='stretch', hide_index=True)
        
        with col2:
            try:
                import plotly.express as px
                
                chart_df = df.head(15).copy()
                fig = px.bar(chart_df, x=metric_names[selected_metric], y='词语', orientation='h',
                           title=f"{metric_names[selected_metric]}排名（前15）",
                           color=metric_names[selected_metric], color_continuous_scale='Blues')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, width='stretch')
            except ImportError:
                st.bar_chart(df.set_index('词语')[metric_names[selected_metric]].head(15))
    
    with st.expander("查看完整中心性数据"):
        full_data = []
        for node, metrics in centrality.items():
            row = {'词语': node}
            for metric_key, metric_name in metric_names.items():
                row[metric_name] = round(metrics.get(metric_key, 0), 4)
            full_data.append(row)
        
        full_df = pd.DataFrame(full_data)
        full_df = full_df.sort_values('度中心性', ascending=False)
        st.dataframe(full_df, width='stretch', hide_index=True)


def _render_export_section(builder):
    """渲染导出部分"""
    import streamlit as st
    
    st.subheader("数据导出")
    
    nodes_csv, edges_csv = builder.export_network()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(label=" 下载节点数据 (CSV)", data=nodes_csv, file_name="semantic_network_nodes.csv", mime="text/csv")
    
    with col2:
        st.download_button(label=" 下载边数据 (CSV)", data=edges_csv, file_name="semantic_network_edges.csv", mime="text/csv")
    
    with st.expander("预览节点数据"):
        if nodes_csv:
            import io
            nodes_df = pd.read_csv(io.StringIO(nodes_csv))
            st.dataframe(nodes_df.head(20), width='stretch')
    
    with st.expander("预览边数据"):
        if edges_csv:
            import io
            edges_df = pd.read_csv(io.StringIO(edges_csv))
            st.dataframe(edges_df.head(20), width='stretch')
