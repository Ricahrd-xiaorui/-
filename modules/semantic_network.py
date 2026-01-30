# -*- coding: utf-8 -*-
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
        
        communities = self.detect_communities()
        if communities:
            stats['num_communities'] = len(set(communities.values()))
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
        
        communities = self.detect_communities()
        centrality = self.calculate_centrality()
        
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
            from modules.frequency_analyzer import CooccurrenceAnalyzer
            
            texts = st.session_state["texts"]
            with st.spinner("正在计算共现关系..."):
                analyzer = CooccurrenceAnalyzer(texts, quick_window_size)
                cooccurrence = analyzer.filter_by_threshold(quick_min_freq)
                
                if cooccurrence:
                    st.session_state["cooccurrence_matrix"] = cooccurrence
                    st.success(f"共现分析完成，找到 {len(cooccurrence)} 对共现词语")
                    st.rerun()
                else:
                    st.warning("未找到符合条件的共现词语对")
        return
    
    texts = st.session_state["texts"]
    
    st.subheader("⚙️ 网络构建参数")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_weight = st.slider("最小边权重", min_value=1, max_value=20, value=2, help="过滤共现频率低于此值的边", key="semantic_min_weight")
    
    with col2:
        max_nodes = st.slider("最大节点数", min_value=20, max_value=200, value=50, help="可视化显示的最大节点数量", key="semantic_max_nodes")
    
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
        
        with st.spinner("正在构建语义网络..."):
            builder = SemanticNetworkBuilder(texts, cooccurrence_data)
            network = builder.build_network(min_weight=min_weight)
            
            if network and network.number_of_nodes() > 0:
                if center_word and center_word in network:
                    filtered_network = builder.filter_by_center(center_word, max_depth)
                    if filtered_network:
                        builder.network = filtered_network
                        st.info(f"已过滤为以「{center_word}」为中心的子网络")
                elif center_word:
                    st.warning(f"核心概念词「{center_word}」不在网络中，显示完整网络")
                
                st.session_state["semantic_network"] = builder.network
                st.session_state["semantic_network_builder"] = builder
                st.session_state["community_labels"] = builder.detect_communities()
                st.session_state["centrality_metrics"] = builder.calculate_centrality()
                
                log_message(f"语义网络构建完成，{builder.network.number_of_nodes()}个节点，{builder.network.number_of_edges()}条边")
                st.success("语义网络构建完成！")
            else:
                st.warning("无法构建语义网络，请检查共现数据或降低最小边权重")
    
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
        
        stats = builder.get_network_statistics()
        
        st.subheader("📊 网络统计")
        stat_cols = st.columns(6)
        stat_cols[0].metric("节点数", stats.get('num_nodes', 0))
        stat_cols[1].metric("边数", stats.get('num_edges', 0))
        stat_cols[2].metric("网络密度", f"{stats.get('density', 0):.4f}")
        stat_cols[3].metric("平均度", f"{stats.get('avg_degree', 0):.2f}")
        stat_cols[4].metric("连通分量", stats.get('num_components', 0))
        stat_cols[5].metric("社区数", stats.get('num_communities', 0))
        
        result_tabs = st.tabs(["🕸️ 网络可视化", "👥 社区分析", "📈 中心性分析", "💾 数据导出"])
        
        with result_tabs[0]:
            _render_network_visualization(builder, max_nodes)
        
        with result_tabs[1]:
            _render_community_analysis(builder, communities)
        
        with result_tabs[2]:
            _render_centrality_analysis(builder, centrality)
        
        with result_tabs[3]:
            _render_export_section(builder)


def _render_network_visualization(builder, max_nodes: int):
    """渲染网络可视化"""
    import streamlit as st
    
    st.subheader("语义网络图")
    
    with st.expander("🎨 可视化设置", expanded=True):
        vis_col1, vis_col2, vis_col3 = st.columns(3)
        
        with vis_col1:
            layout_algorithm = st.selectbox("布局算法", ["spring (力导向)", "kamada_kawai", "circular", "shell", "spectral"], index=0, key="semantic_layout_algo")
        
        with vis_col2:
            color_by = st.selectbox("节点颜色依据", ["社区", "度中心性", "介数中心性"], index=0, key="semantic_color_by")
        
        with vis_col3:
            show_labels = st.checkbox("显示节点标签", value=True, key="semantic_show_labels")
    
    nodes, edges = builder.to_vis_data(max_nodes)
    
    if not nodes:
        st.warning("没有可显示的节点")
        return
    
    try:
        import plotly.graph_objects as go
        
        layout_name = layout_algorithm.split()[0]
        pos = _get_network_layout(builder.network, layout_name, max_nodes)
        
        if not pos:
            st.warning("无法计算网络布局")
            return
        
        communities = builder.detect_communities()
        centrality = builder.calculate_centrality()
        
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        
        for node_data in nodes:
            node_id = node_data['id']
            if node_id in pos:
                node_x.append(pos[node_id][0])
                node_y.append(pos[node_id][1])
                
                degree = node_data.get('size', 1)
                community = communities.get(node_id, 0)
                node_text.append(f"{node_id}<br>度数: {degree}<br>社区: {community}")
                
                if color_by == "社区":
                    node_color.append(community)
                elif color_by == "度中心性":
                    node_color.append(centrality.get(node_id, {}).get('degree', 0))
                else:
                    node_color.append(centrality.get(node_id, {}).get('betweenness', 0))
                
                node_size.append(max(10, min(50, degree * 3)))
        
        edge_x, edge_y = [], []
        
        for edge in edges:
            source, target = edge['source'], edge['target']
            if source in pos and target in pos:
                edge_x.extend([pos[source][0], pos[target][0], None])
                edge_y.extend([pos[source][1], pos[target][1], None])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none', name='边'))
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(size=node_size, color=node_color, colorscale='Viridis', showscale=True, colorbar=dict(title=color_by)),
            text=[n['label'] for n in nodes if n['id'] in pos] if show_labels else None,
            textposition='top center', textfont=dict(size=9),
            hovertext=node_text, hoverinfo='text', name='节点'
        ))
        
        fig.update_layout(
            title=f"语义网络图 ({len(nodes)}个节点, {len(edges)}条边)",
            showlegend=False, hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except ImportError:
        st.warning("需要安装plotly库才能显示网络图: pip install plotly")
        st.markdown("**网络节点列表（按度数排序）：**")
        node_df = pd.DataFrame(nodes)
        st.dataframe(node_df, use_container_width=True)


def _get_network_layout(network, layout_name: str, max_nodes: int) -> Dict[str, Tuple[float, float]]:
    """获取网络布局"""
    if not HAS_NETWORKX or network is None:
        return {}
    
    try:
        if network.number_of_nodes() > max_nodes:
            node_degrees = dict(network.degree())
            top_nodes = sorted(node_degrees.items(), key=lambda x: -x[1])[:max_nodes]
            top_node_ids = [n for n, _ in top_nodes]
            subgraph = network.subgraph(top_node_ids)
        else:
            subgraph = network
        
        if layout_name == "spring":
            pos = nx.spring_layout(subgraph, k=2, iterations=50)
        elif layout_name == "kamada_kawai":
            pos = nx.kamada_kawai_layout(subgraph)
        elif layout_name == "circular":
            pos = nx.circular_layout(subgraph)
        elif layout_name == "shell":
            pos = nx.shell_layout(subgraph)
        elif layout_name == "spectral":
            try:
                pos = nx.spectral_layout(subgraph)
            except Exception:
                pos = nx.spring_layout(subgraph)
        else:
            pos = nx.spring_layout(subgraph)
        
        return pos
    except Exception:
        return {}


def _render_community_analysis(builder, communities: Dict[str, int]):
    """渲染社区分析"""
    import streamlit as st
    
    st.subheader("社区检测结果")
    
    if not communities:
        st.info("没有检测到社区")
        return
    
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
                st.dataframe(member_df, use_container_width=True, hide_index=True)
            else:
                st.write(", ".join(members[:20]))
                if len(members) > 20:
                    st.write(f"... 等共 {len(members)} 个词语")


def _render_centrality_analysis(builder, centrality: Dict[str, Dict[str, float]]):
    """渲染中心性分析"""
    import streamlit as st
    
    st.subheader("中心性指标分析")
    
    if not centrality:
        st.info("没有中心性数据")
        return
    
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
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        with col2:
            try:
                import plotly.express as px
                
                chart_df = df.head(15).copy()
                fig = px.bar(chart_df, x=metric_names[selected_metric], y='词语', orientation='h',
                           title=f"{metric_names[selected_metric]}排名（前15）",
                           color=metric_names[selected_metric], color_continuous_scale='Blues')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
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
        st.dataframe(full_df, use_container_width=True, hide_index=True)


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
            st.dataframe(nodes_df.head(20), use_container_width=True)
    
    with st.expander("预览边数据"):
        if edges_csv:
            import io
            edges_df = pd.read_csv(io.StringIO(edges_csv))
            st.dataframe(edges_df.head(20), use_container_width=True)
