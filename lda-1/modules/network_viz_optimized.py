# -*- coding: utf-8 -*-
"""
优化的网络图可视化模块

专门用于高性能渲染共现网络图
"""

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from typing import List, Dict, Tuple


def render_optimized_network(nodes: List[dict], edges: List[dict], 
                            layout_algorithm: str = "spring",
                            color_scheme: str = "学术蓝",
                            show_labels: bool = True,
                            enable_community: bool = False) -> go.Figure:
    """
    优化的网络图渲染函数
    
    Args:
        nodes: 节点列表
        edges: 边列表
        layout_algorithm: 布局算法名称
        color_scheme: 配色方案
        show_labels: 是否显示标签
        enable_community: 是否启用社区检测
    
    Returns:
        Plotly Figure 对象
    """
    
    # 构建NetworkX图
    G = nx.Graph()
    for node in nodes:
        G.add_node(node["id"], size=node["size"])
    for edge in edges:
        G.add_edge(edge["source"], edge["target"], weight=edge["weight"])
    
    # 计算布局（优化迭代次数）
    max_iterations = 50 if len(nodes) > 50 else 100
    
    layout_name = layout_algorithm.split(" ")[0]
    
    if layout_name == "spring":
        pos = nx.spring_layout(G, k=2, iterations=max_iterations, seed=42)
    elif layout_name == "circular":
        pos = nx.circular_layout(G)
    elif layout_name == "random":
        pos = nx.random_layout(G, seed=42)
    elif layout_name == "shell":
        pos = nx.shell_layout(G)
    else:
        # 默认使用 spring
        pos = nx.spring_layout(G, k=2, iterations=max_iterations, seed=42)
    
    # 配色方案
    color_schemes = {
        "学术蓝": {
            "edge_color": "#90A4AE",
            "node_colorscale": "Blues",
            "bg_color": "white"
        },
        "经典灰": {
            "edge_color": "#B0BEC5",
            "node_colorscale": "Greys",
            "bg_color": "white"
        },
        "暖色调": {
            "edge_color": "#FFCC80",
            "node_colorscale": "Oranges",
            "bg_color": "white"
        },
        "冷色调": {
            "edge_color": "#A5D6A7",
            "node_colorscale": "Greens",
            "bg_color": "white"
        }
    }
    
    scheme = color_schemes.get(color_scheme, color_schemes["学术蓝"])
    
    # 优化：将所有边合并到一个 trace 中
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color=scheme["edge_color"]),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # 节点数据
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    node_degrees = dict(G.degree())
    max_degree = max(node_degrees.values()) if node_degrees else 1
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node if show_labels else "")
        
        degree = node_degrees[node]
        node_size.append(15 + (degree / max_degree) * 35)
        node_color.append(degree)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text' if show_labels else 'markers',
        text=node_text,
        textposition='top center',
        textfont=dict(size=10),
        hoverinfo='text',
        hovertext=[f"{node}<br>连接数: {node_degrees[node]}" for node in G.nodes()],
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale=scheme["node_colorscale"],
            showscale=True,
            colorbar=dict(title="连接数", thickness=15),
            line=dict(width=1, color='white')
        )
    )
    
    # 创建图形
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"词语共现网络图 ({len(nodes)}个节点, {len(edges)}条边)",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor=scheme["bg_color"],
            height=600
        )
    )
    
    return fig
