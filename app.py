# -*- coding: utf-8 -*-
"""
文件可视化分析系统 - 主入口

UI结构：
0. 用户认证 - 登录注册
1. 数据加载 - 文件上传、预览
2. 文本预处理 - 词典管理、分词、停用词
3. 基础文本分析 - 文本统计、词频分析、词语共现
4. 主题建模 - LDA训练、最优主题搜索
5. 主题可视化 - 词云、热图、PyLDAvis等
6. 高级研究分析 - 聚类、时序、比较、引用、语义网络、质性编码
7. 结果导出 - 导出各类分析结果
"""

import importlib
import importlib.util
import sys

REQUIRED_PACKAGES = {
    'jieba': '中文分词',
    'gensim': '主题建模',
    'networkx': '网络分析',
    'plotly': '交互图表',
    'pdfplumber': 'PDF读取',
    'python-docx': 'Word读取(.docx)',
    'pyyaml': '配置解析',
    'pandas': '数据处理',
    'numpy': '数值计算',
    'scikit-learn': '机器学习',
}

OPTIONAL_PACKAGES = {
    'wordcloud': '词云生成',
    'PyPDF2': 'PDF备用读取',
    'community': '社区检测(Louvain)',
    'matplotlib': '图表绘制',
    'seaborn': '统计绘图',
}

REQUIRED_IMPORT_NAMES = {
    'pyyaml': 'yaml',
    'scikit-learn': 'sklearn',
    'python-docx': 'docx',
}

def check_dependencies():
    """检查并报告依赖库状态"""
    missing_required = []
    missing_optional = []
    
    for package, name in REQUIRED_PACKAGES.items():
        import_name = REQUIRED_IMPORT_NAMES.get(package, package)
        if importlib.util.find_spec(import_name) is None:
            missing_required.append(f"{name} ({package})")
    
    for package, name in OPTIONAL_PACKAGES.items():
        import_name = REQUIRED_IMPORT_NAMES.get(package, package)
        if importlib.util.find_spec(import_name) is None:
            missing_optional.append(f"{name} ({package})")
    
    return missing_required, missing_optional

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from pathlib import Path

_missing_required, _missing_optional = check_dependencies()

if _missing_required:
    st.error(f"缺少必需依赖库: {', '.join(_missing_required)}。请运行: pip install {' '.join(REQUIRED_PACKAGES.keys())}")
    st.stop()

if _missing_optional:
    st.warning(f"缺少可选依赖库: {', '.join(_missing_optional)}。部分功能可能不可用。")

# 配置matplotlib中文字体
from utils.font_config import setup_matplotlib_chinese
setup_matplotlib_chinese()

# 导入用户认证模块
from modules.auth import check_authentication, render_user_management, check_permission

# 导入核心模块
from modules.sidebar import render_system_sidebar
from modules.data_loader import render_data_loader
from modules.text_processor import render_text_processor
from modules.lda_trainer import render_model_trainer
from modules.topic_visualization import render_visualizer
from modules.result_exporter import render_exporter
from utils.session_state import get_session_state, initialize_session_state
from config.system import ensure_directories

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
render_text_statistics = safe_import_render_function('text_analysis_readability', 'render_text_statistics')
render_dictionary_manager = safe_import_render_function('dictionary_manager', 'render_dictionary_manager')

# 导入待实现的模块
render_frequency_analyzer = safe_import_render_function('word_frequency_cooccurrence', 'render_frequency_analyzer')
render_clustering_module = safe_import_render_function('clustering_module', 'render_clustering_module')
render_temporal_analyzer = safe_import_render_function('temporal_topic_evolution', 'render_temporal_analyzer')
render_comparative_analyzer = safe_import_render_function('comparative_analyzer', 'render_comparative_analyzer')
render_citation_analyzer = safe_import_render_function('citation_analyzer', 'render_citation_analyzer')
render_semantic_network = safe_import_render_function('semantic_network_builder', 'render_semantic_network')
render_qualitative_coding = safe_import_render_function('qualitative_coding', 'render_qualitative_coding')

# 页面配置
st.set_page_config(
    page_title="文件可视化分析系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def local_css():
    """应用自定义CSS样式"""
    css = """
    <style>
    /* 全局样式 */
    .main {
        padding: 0 20px;
    }
    
    /* 标题样式 */
    .main-header {
        text-align: center;
        padding: 20px 0;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 10px 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    /* 工作流程卡片 */
    .workflow-card {
        background: white;
        border-radius: 10px;
        padding: 15px 20px;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #2C3E50;
        transition: all 0.3s ease;
    }
    .workflow-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
    }
    .workflow-card.completed {
        border-left-color: #27AE60;
        background: #f0f9f4;
    }
    .workflow-card.active {
        border-left-color: #3498DB;
        background: #EBF5FB;
    }
    .workflow-card.pending {
        border-left-color: #BDC3C7;
        opacity: 0.7;
    }
    
    /* Tab 样式优化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #EDF2F7;
        padding: 8px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.5);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: white !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* 进度条样式 */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3498DB, #667eea);
        border-radius: 10px;
    }
    
    /* 成功/信息/警告框美化 */
    .success-box, .info-box, .warning-box {
        padding: 15px 20px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    .success-box {
        background: #E8F8F5;
        border-color: #27AE60;
    }
    .info-box {
        background: #EBF5FB;
        border-color: #3498DB;
    }
    .warning-box {
        background: #FEF9E7;
        border-color: #F39C12;
    }
    
    /* 指标卡片 */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #EDF2F7;
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2C3E50;
    }
    .metric-card .metric-label {
        font-size: 0.9rem;
        color: #7F8C8D;
        margin-top: 5px;
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background: #2C3E50;
    }
    
    /* Expander 美化 */
    .streamlit-expander {
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    
    /* 按钮样式 */
    .stButton > button {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* 数据框样式 */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* 页脚样式 */
    .footer {
        text-align: center;
        padding: 30px 0;
        margin-top: 40px;
        border-top: 1px solid #E2E8F0;
        color: #7F8C8D;
    }
    .footer a {
        color: #3498DB;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* 动画效果 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-fadeIn {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* 加载动画 */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(52, 152, 219, 0.3);
        border-radius: 50%;
        border-top-color: #3498DB;
        animation: spin 1s ease-in-out infinite;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_workflow_indicator():
    """渲染工作流程步骤指示器"""
    steps = [
        ("📂", "数据加载", bool(st.session_state.get("raw_texts"))),
        ("✏️", "文本预处理", bool(st.session_state.get("texts") and st.session_state.get("corpus"))),
        ("📈", "基础分析", bool(st.session_state.get("texts"))),
        ("🧠", "主题建模", bool(st.session_state.get("training_complete"))),
        ("🎨", "可视化", bool(st.session_state.get("training_complete"))),
        ("📦", "结果导出", bool(st.session_state.get("training_complete")))
    ]
    
    status_colors = {
        "completed": "#27AE60",
        "active": "#3498DB",
        "pending": "#95A5A6"
    }
    
    cols = st.columns(len(steps))
    for i, (icon, name, completed) in enumerate(steps):
        with cols[i]:
            if completed:
                status = "completed"
                status_icon = "✓"
                status_text = "已完成"
                bg_color = "#f0f9f4"
            elif any(s[2] for s in steps[:i]):
                status = "active"
                status_icon = "▶"
                status_text = "进行中"
                bg_color = "#EBF5FB"
            else:
                status = "pending"
                status_icon = "○"
                status_text = "待处理"
                bg_color = "#FAFBFC"
            
            st.markdown(f'<div style="background: {bg_color}; border-radius: 10px; padding: 15px 10px; text-align: center; border-left: 4px solid {status_colors[status]}; box-shadow: 0 2px 8px rgba(0,0,0,0.06);"><div style="font-size: 1.5rem; margin-bottom: 5px;">{icon}</div><div style="font-weight: 600; font-size: 0.85rem; color: #2C3E50;">{name}</div><div style="font-size: 0.7rem; color: {status_colors[status]}; margin-top: 3px;">{status_icon} {status_text}</div></div>', unsafe_allow_html=True)


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
            from modules.word_frequency_cooccurrence import render_cooccurrence_analyzer
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
    check_authentication()
    
    local_css()
    initialize_session_state()
    ensure_directories()
    
    # 检查用户角色
    user_info = st.session_state.get("user_info", {})
    is_admin = user_info.get("role") == "admin"
    
    # 标题区域
    st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3); margin-bottom: 20px;"><h1 style="margin: 0; font-size: 2rem; font-weight: 700;">📊 文件可视化分析系统</h1><p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 1rem;">基于LDA主题建模的智能文本分析平台 | 支持词云、热力图、时序分析等多种可视化</p></div>', unsafe_allow_html=True)
    
    render_system_sidebar()
    render_workflow_indicator()
    
    tab_configs = [
        ("📂 数据加载", render_data_loader, "data_load"),
        ("✏️ 文本预处理", render_text_processor, "text_process"),
        ("📈 基础文本分析", render_basic_text_analysis, "basic_analysis"),
        ("🧠 主题建模", render_model_trainer, "topic_modeling"),
        ("🎨 主题可视化", render_topic_visualization, "visualization"),
        ("🔬 高级研究分析", render_advanced_analysis, "advanced_analysis"),
        ("📦 结果导出", render_exporter, "export"),
    ]

    available_tabs = []
    for label, render_func, permission in tab_configs:
        if check_permission(permission):
            available_tabs.append((label, render_func))

    if is_admin:
        available_tabs.append(("👥 用户管理", render_user_management))

    if not available_tabs:
        st.error("您没有任何功能权限，请联系管理员")
        st.stop()

    tab_labels = [label for label, _ in available_tabs]
    tab_renderers = [render_func for _, render_func in available_tabs]

    main_tabs = st.tabs(tab_labels)

    for idx, render_func in enumerate(tab_renderers):
        with main_tabs[idx]:
            render_func()
    
    # 页脚
    st.markdown('<div style="text-align: center; padding: 30px 0; margin-top: 40px; border-top: 1px solid #E2E8F0; color: #7F8C8D;"><p style="margin: 0;"><strong>📊 文件可视化分析系统</strong> v2.0.0</p><p style="margin: 5px 0 0 0;">基于 Streamlit + Gensim + PyLDAvis 构建</p><p style="margin: 10px 0 0 0; font-size: 0.8rem;">支持 LDA 主题建模 | 词云可视化 | 时序分析 | 语义网络 | 质性编码</p></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
