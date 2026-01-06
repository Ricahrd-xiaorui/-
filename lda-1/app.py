import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from pathlib import Path

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 导入自定义模块
from modules.sidebar import render_system_sidebar
from modules.data_loader import render_data_loader
from modules.text_processor import render_text_processor
from modules.model_trainer import render_model_trainer
from modules.visualizer import render_visualizer
from modules.exporter import render_exporter
from utils.session_state import get_session_state, initialize_session_state

# 页面配置
st.set_page_config(
    page_title="政策文件LDA主题模型分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用CSS
def local_css():
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
        ("模型训练", bool(st.session_state.get("training_complete"))),
        ("可视化", bool(st.session_state.get("training_complete"))),
        ("导出", bool(st.session_state.get("training_complete")))
    ]
    
    # 构建步骤显示
    cols = st.columns(len(steps))
    for i, (name, completed) in enumerate(steps):
        with cols[i]:
            if completed:
                st.success(f"✅ {name}")
            elif i == 0 or steps[i-1][1]:
                st.info(f"👉 {name}")
            else:
                st.empty()

def main():
    # 应用CSS
    local_css()
    
    # 初始化会话状态
    initialize_session_state()
    
    # 标题
    st.title("政策文件LDA主题模型可视化分析系统")
    
    # 创建基本目录结构
    Path("temp").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # 系统侧边栏（只包含系统状态、日志和帮助）
    render_system_sidebar()
    
    # 工作流程步骤指示器
    render_workflow_indicator()
    
    # 创建标签页
    tabs = st.tabs(["📁 数据加载", "⚙️ 文本预处理", "🎯 模型训练", "📊 可视化分析", "💾 结果导出"])
    
    # 数据加载标签页
    with tabs[0]:
        render_data_loader()
    
    # 文本预处理标签页
    with tabs[1]:
        render_text_processor()
    
    # 模型训练标签页
    with tabs[2]:
        render_model_trainer()
    
    # 可视化分析标签页
    with tabs[3]:
        render_visualizer()
    
    # 结果导出标签页
    with tabs[4]:
        render_exporter()
    
    # 显示页脚
    st.markdown("---")
    st.caption("政策文件LDA主题模型可视化分析系统 | 版本 1.0.0")

if __name__ == "__main__":
    main() 