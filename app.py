# -*- coding: utf-8 -*-
"""
文件可视化分析系统 - 主入口

UI结构：
1. 数据加载 - 文件上传、预览
2. 文本预处理 - 词典管理、分词、停用词
3. 基础文本分析 - 文本统计、词频分析、词语共现
4. 主题建模 - LDA训练、最优主题搜索
5. 主题可视化 - 词云、热图、PyLDAvis等
6. 高级研究分析 - 聚类、时序、比较、引用、语义网络、质性编码
7. 结果导出 - 导出各类分析结果
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from pathlib import Path

# 配置matplotlib中文字体
from utils.font_config import setup_matplotlib_chinese
setup_matplotlib_chinese()

# 导入核心模块
from modules.sidebar import render_system_sidebar
from modules.data_loader import render_data_loader
from modules.text_processor import render_text_processor
from modules.model_trainer import render_model_trainer
from modules.visualizer import render_visualizer
from modules.exporter import render_exporter
from utils.session_state import get_session_state, initialize_session_state

# 安全导入渲染函数（模块不存在时返回占位函数）
def safe_import_render_function(module_name, function_name):
    """安全导入渲染函数，模块不存在时返回占位函数"""
    try:
        module = __import__(f'modules.{module_name}', fromlist=[function_name])
        return getattr(module, function_name)
    except (ImportError, AttributeError):
        def placeholder():
            st.info(f"📦 {module_name} 模块正在开发中...")
        return placeholder

# 导入已实现的模块
render_text_statistics = safe_import_render_function('text_statistics', 'render_text_statistics')
render_dictionary_manager = safe_import_render_function('dictionary_manager', 'render_dictionary_manager')

# 导入待实现的模块
render_frequency_analyzer = safe_import_render_function('frequency_analyzer', 'render_frequency_analyzer')
render_clustering_module = safe_import_render_function('clustering_module', 'render_clustering_module')
render_temporal_analyzer = safe_import_render_function('temporal_analyzer', 'render_temporal_analyzer')
render_comparative_analyzer = safe_import_render_function('comparative_analyzer', 'render_comparative_analyzer')
render_citation_analyzer = safe_import_render_function('citation_analyzer', 'render_citation_analyzer')
render_semantic_network = safe_import_render_function('semantic_network', 'render_semantic_network')
render_qualitative_coding = safe_import_render_function('qualitative_coding', 'render_qualitative_coding')

# 页面配置
st.set_page_config(
    page_title="文件可视化分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def local_css():
    """应用自定义CSS样式"""
    css = """
    <style>
    .main {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        border-radius: 4px 4px 0px 0px;
    }
    .stProgress > div > div {
        background-color: #1E88E5;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_workflow_indicator():
    """渲染工作流程步骤指示器"""
    steps = [
        ("数据加载", bool(st.session_state.get("raw_texts"))),
        ("文本预处理", bool(st.session_state.get("texts") and st.session_state.get("corpus"))),
        ("基础分析", bool(st.session_state.get("texts"))),
        ("主题建模", bool(st.session_state.get("training_complete"))),
        ("可视化", bool(st.session_state.get("training_complete"))),
        ("导出", bool(st.session_state.get("training_complete")))
    ]
    
    cols = st.columns(len(steps))
    for i, (name, completed) in enumerate(steps):
        with cols[i]:
            if completed:
                st.success(f"✅ {name}")
            elif i == 0 or steps[i-1][1]:
                st.info(f"👉 {name}")
            else:
                st.empty()


def render_basic_text_analysis():
    """渲染基础文本分析模块"""
    st.header("📈 基础文本分析")
    
    # 检查是否有数据
    if not st.session_state.get("raw_texts"):
        st.warning("⚠️ 请先在「数据加载」标签页中加载文本数据")
        return
    
    # 检查是否完成预处理
    if not st.session_state.get("texts"):
        st.warning("⚠️ 请先在「文本预处理」标签页中完成文本预处理")
        return
    
    # 创建子标签页
    analysis_tabs = st.tabs([
        "📊 文本统计",
        "🔢 词频分析", 
        "🔗 词语共现"
    ])
    
    # 文本统计
    with analysis_tabs[0]:
        render_text_statistics()
    
    # 词频分析
    with analysis_tabs[1]:
        render_frequency_analyzer()
    
    # 词语共现
    with analysis_tabs[2]:
        # 如果词频分析模块包含共现功能，可以在这里调用
        # 否则显示占位信息
        try:
            from modules.frequency_analyzer import render_cooccurrence_analyzer
            render_cooccurrence_analyzer()
        except (ImportError, AttributeError):
            st.info("📦 词语共现分析模块正在开发中...")


def render_topic_visualization():
    """渲染主题可视化模块（精简版，移除文档聚类）"""
    # 直接调用原有的可视化模块
    render_visualizer()


def render_advanced_analysis():
    """渲染高级研究分析模块"""
    st.header("🔬 高级研究分析")
    st.markdown("面向学术研究的高级文本分析功能")
    
    # 检查是否有数据
    if not st.session_state.get("raw_texts"):
        st.warning("⚠️ 请先在「数据加载」标签页中加载文本数据")
        return
    
    # 创建子标签页
    advanced_tabs = st.tabs([
        "🎯 聚类分类",
        "📅 时序分析",
        "🔍 比较分析",
        "📖 引用分析",
        "🕸️ 语义网络",
        "🏷️ 质性编码"
    ])
    
    # 聚类分类
    with advanced_tabs[0]:
        render_clustering_module()
    
    # 时序分析
    with advanced_tabs[1]:
        render_temporal_analyzer()
    
    # 比较分析
    with advanced_tabs[2]:
        render_comparative_analyzer()
    
    # 引用分析
    with advanced_tabs[3]:
        render_citation_analyzer()
    
    # 语义网络
    with advanced_tabs[4]:
        render_semantic_network()
    
    # 质性编码
    with advanced_tabs[5]:
        render_qualitative_coding()


def main():
    """主函数"""
    # 应用CSS
    local_css()
    
    # 初始化会话状态
    initialize_session_state()
    
    # 标题
    st.title("📊 文件可视化分析系统")
    
    # 创建基本目录结构
    Path("temp").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # 系统侧边栏
    render_system_sidebar()
    
    # 工作流程步骤指示器
    render_workflow_indicator()
    
    # 创建主标签页
    main_tabs = st.tabs([
        "📁 数据加载", 
        "⚙️ 文本预处理", 
        "📈 基础文本分析",
        "🎯 主题建模", 
        "📊 主题可视化", 
        "🔬 高级研究分析",
        "💾 结果导出"
    ])
    
    # 1. 数据加载
    with main_tabs[0]:
        render_data_loader()
    
    # 2. 文本预处理（含词典管理）
    with main_tabs[1]:
        render_text_processor()
    
    # 3. 基础文本分析
    with main_tabs[2]:
        render_basic_text_analysis()
    
    # 4. 主题建模
    with main_tabs[3]:
        render_model_trainer()
    
    # 5. 主题可视化
    with main_tabs[4]:
        render_topic_visualization()
    
    # 6. 高级研究分析
    with main_tabs[5]:
        render_advanced_analysis()
    
    # 7. 结果导出
    with main_tabs[6]:
        render_exporter()
    
    # 页脚
    st.markdown("---")
    st.caption("文件可视化分析系统 | 版本 2.0.0 |")


if __name__ == "__main__":
    main()
