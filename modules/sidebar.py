import streamlit as st
import os
import time
from datetime import datetime
from utils.session_state import save_session_state, load_session_state, log_message

# 侧边栏自定义CSS
def sidebar_css():
    st.markdown("""
    <style>
    /* 侧边栏样式 */
    .sidebar-section {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
    }
    .sidebar-title {
        font-size: 1rem;
        font-weight: 600;
        color: #2C3E50;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 2px solid #3498DB;
    }
    .sidebar-metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px dashed #E2E8F0;
    }
    .sidebar-metric:last-child {
        border-bottom: none;
    }
    .sidebar-metric-label {
        color: #7F8C8D;
        font-size: 0.85rem;
    }
    .sidebar-metric-value {
        color: #2C3E50;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* 侧边栏按钮样式 */
    .sidebar-btn {
        width: 100%;
        padding: 10px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .sidebar-btn-primary {
        background: linear-gradient(135deg, #3498DB, #2980B9);
        color: white;
    }
    .sidebar-btn-primary:hover {
        background: linear-gradient(135deg, #2980B9, #2471A3);
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(52, 152, 219, 0.3);
    }
    .sidebar-btn-secondary {
        background: #EDF2F7;
        color: #2C3E50;
    }
    .sidebar-btn-secondary:hover {
        background: #E2E8F0;
    }
    .sidebar-btn-danger {
        background: linear-gradient(135deg, #E74C3C, #C0392B);
        color: white;
    }
    .sidebar-btn-danger:hover {
        background: linear-gradient(135deg, #C0392B, #A93226);
    }
    
    /* 缓存信息样式 */
    .cache-item {
        display: flex;
        align-items: center;
        padding: 6px 10px;
        background: #F8F9FA;
        border-radius: 6px;
        margin: 5px 0;
        font-size: 0.85rem;
    }
    .cache-item::before {
        content: "•";
        color: #3498DB;
        font-size: 1.2rem;
        margin-right: 8px;
    }
    
    /* 工作流程指示器样式 */
    .step-indicator {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        background: #EBF5FB;
        border-radius: 6px;
        margin: 5px 0;
        border-left: 3px solid #3498DB;
    }
    .step-indicator.completed {
        background: #E8F8F5;
        border-left-color: #27AE60;
    }
    .step-indicator.pending {
        background: #F8F9FA;
        border-left-color: #BDC3C7;
    }
    </style>
    """, unsafe_allow_html=True)

def render_system_status():
    """渲染系统状态区域"""
    sidebar_css()
    
    with st.sidebar.expander("⚙️ 系统状态", expanded=True):
        if st.session_state.get("current_step"):
            st.info(f"当前步骤: {st.session_state['current_step']}")
            st.progress(st.session_state["progress"])
        
        doc_count = len(st.session_state.get("raw_texts", []))
        if doc_count > 0:
            st.metric("📄 已加载文档", f"{doc_count} 个")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 保存会话", key="save_session_action", use_container_width=True):
                save_path = os.path.join("models", f"session_{int(time.time())}.json")
                save_session_state(save_path)
                st.success(f"会话已保存!")
                log_message(f"会话状态已保存到: {save_path}")

        with col2:
            if st.button("📂 加载会话", key="load_session_action", use_container_width=True):
                st.session_state["show_load_session"] = True

@st.fragment(run_every=1)
def render_realtime_clock():
    """实时显示运行时间（每秒自动更新）"""
    if "start_time" not in st.session_state:
        st.session_state["start_time"] = time.time()
    
    elapsed_time = time.time() - st.session_state["start_time"]
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    if hours > 0:
        time_str = f"{hours}时 {minutes}分 {seconds}秒"
    else:
        time_str = f"{minutes}分 {seconds}秒"
    
    st.metric("⏱️ 运行时间", time_str)

def render_load_session_dialog():
    """渲染加载会话对话框"""
    if st.session_state.get("show_load_session", False):
        with st.sidebar.expander("加载会话", expanded=True):
            models_dir = "models"
            os.makedirs(models_dir, exist_ok=True)
            session_files = [f for f in os.listdir(models_dir) if f.endswith(".json")] if os.path.exists(models_dir) else []
            if session_files:
                session_files.sort(reverse=True)
                selected_file = st.selectbox("选择会话文件", session_files, key="session_file_select")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("确认加载", key="confirm_load_session"):
                        load_path = os.path.join(models_dir, selected_file)
                        if load_session_state(load_path):
                            st.success("会话加载成功")
                            log_message(f"会话已从 {selected_file} 加载", level="success")
                            st.session_state["show_load_session"] = False
                            st.rerun()
                with col2:
                    if st.button("取消", key="cancel_load_session"):
                        st.session_state["show_load_session"] = False
                        st.rerun()
            else:
                st.warning("未找到可用的会话文件，请先保存会话")

# 已移除render_analysis_config()和render_visualization_options()函数
# 这些配置已移至各自的功能页面：
# - 分析配置在"模型训练"页面
# - 可视化选项在"可视化分析"页面

def render_log_area():
    """渲染日志区域"""
    with st.sidebar.expander("📝 系统日志", expanded=False):
        if st.session_state.get("log_messages"):
            log_container = st.container()
            with log_container:
                logs = st.session_state["log_messages"]
                display_logs = logs[-50:] if len(logs) > 50 else logs
                for log in reversed(display_logs):
                    message = f"{log['time']} - {log['message']}"
                    if log['level'] == 'info':
                        st.text(message)
                    elif log['level'] == 'warning':
                        st.warning(message)
                    elif log['level'] == 'error':
                        st.error(message)
                    elif log['level'] == 'success':
                        st.success(message)
            
            total_logs = len(st.session_state["log_messages"])
            st.caption(f"📊 共 {total_logs} 条日志" if total_logs <= 50 else f"📊 显示最近50条（共{total_logs}条）")
        else:
            st.text("暂无日志记录")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空", key="log_action_clear", use_container_width=True):
                st.session_state["log_messages"] = []
                st.rerun()

        with col2:
            if st.button("📥 导出", key="log_action_export", use_container_width=True):
                export_logs()

def export_logs():
    """导出日志功能"""
    if not st.session_state.get("log_messages"):
        st.warning("没有可导出的日志")
        return
    
    log_text = "\n".join([
        f"{log['time']} - [{log['level'].upper()}] {log['message']}" 
        for log in st.session_state["log_messages"]
    ])
    
    # 创建日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"app_{timestamp}.log")
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)
    
    # 提供下载链接
    with open(log_path, "rb") as f:
        log_data = f.read()
    
    st.download_button(
        label="下载日志文件",
        data=log_data,
        file_name=f"app_{timestamp}.log",
        mime="text/plain",
        key="log_download_btn"
    )
    
    log_message(f"日志已导出到: {log_path}", level="success")

def render_help_section():
    """渲染帮助和关于部分"""
    with st.sidebar.expander("💡 帮助与关于", expanded=False):
        st.markdown('<div style="padding: 10px 0;"><h4 style="color: #2C3E50; margin-bottom: 10px;">📊 LDA主题建模分析系统</h4><p style="font-size: 0.85rem; color: #555; line-height: 1.6;">基于 Streamlit + Gensim 的智能文本主题建模与可视化分析平台。</p></div>', unsafe_allow_html=True)
        
        st.markdown("**🚀 快速使用流程：**")
        st.markdown("1. 📂 **数据加载** - 上传TXT/MD文档\n2. ✏️ **文本预处理** - 分词与去停用词\n3. 🧠 **模型训练** - 训练LDA主题模型\n4. 🎨 **可视化** - 生成词云与热力图\n5. 📦 **结果导出** - 导出分析报告")
        
        st.markdown('<div style="background: #EBF5FB; padding: 10px; border-radius: 8px; margin: 10px 0;"><b>💡 小贴士</b><ul style="font-size: 0.8rem; margin: 5px 0; padding-left: 15px;"><li>每个功能页面都有详细说明</li><li>文档数量建议 3-100 个</li><li>主题数量建议 3-15 个</li></ul></div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align: center; padding: 15px 0; border-top: 1px solid #E2E8F0; margin-top: 10px;"><b style="color: #2C3E50;">版本信息</b><br><span style="font-size: 0.8rem; color: #7F8C8D;">v2.0.0 | 更新: 2026-01-12<br>构建于 Streamlit + Gensim + PyLDAvis</span></div>', unsafe_allow_html=True)

def render_cache_manager():
    """渲染缓存管理区域"""
    with st.sidebar.expander("🗑️ 缓存管理", expanded=False):
        st.markdown("清除系统缓存以释放内存或重新开始分析。")
        
        cache_info = []
        if st.session_state.get("raw_texts"):
            cache_info.append(f"原始文本: {len(st.session_state['raw_texts'])} 个")
        if st.session_state.get("texts"):
            cache_info.append(f"预处理文本: {len(st.session_state['texts'])} 个")
        if st.session_state.get("lda_model"):
            cache_info.append(f"LDA模型: 已加载")
        if st.session_state.get("pyldavis_html"):
            cache_info.append(f"PyLDAvis缓存: 已生成")
        if st.session_state.get("wordcloud_images"):
            cache_info.append(f"词云缓存: {len(st.session_state['wordcloud_images'])} 个")
        
        if cache_info:
            for info in cache_info:
                st.markdown(f'<div style="display: flex; align-items: center; padding: 6px 10px; background: #F8F9FA; border-radius: 6px; margin: 5px 0; font-size: 0.85rem;"><span style="color: #3498DB; margin-right: 8px;">•</span>{info}</div>', unsafe_allow_html=True)
        else:
            st.text("当前无缓存数据")
        
        st.markdown("---")
        
        clear_options = st.multiselect(
            "选择要清除的缓存",
            ["数据缓存", "模型缓存", "可视化缓存", "日志"],
            key="cache_clear_options"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 清除选中", key="cache_clear_selected", use_container_width=True):
                if "数据缓存" in clear_options:
                    st.session_state.file_contents = {}
                    st.session_state.file_names = []
                    st.session_state.raw_texts = []
                    st.session_state.texts = None
                    st.session_state.dictionary = None
                    st.session_state.corpus = None
                    log_message("已清除数据缓存", level="warning")
                
                if "模型缓存" in clear_options:
                    st.session_state.lda_model = None
                    st.session_state.training_complete = False
                    st.session_state.coherence_score = None
                    st.session_state.perplexity = None
                    st.session_state.topic_keywords = {}
                    st.session_state.doc_topic_dist = None
                    st.session_state.model_path = None
                    st.session_state.optimal_search_results = None
                    log_message("已清除模型缓存", level="warning")
                
                if "可视化缓存" in clear_options:
                    st.session_state.pyldavis_html = None
                    st.session_state.wordcloud_images = {}
                    keys_to_remove = [k for k in list(st.session_state.keys()) 
                                     if k.startswith('tsne_') or k.startswith('umap_')]
                    for key in keys_to_remove:
                        del st.session_state[key]
                    log_message("已清除可视化缓存", level="warning")
                
                if "日志" in clear_options:
                    st.session_state.log_messages = []
                    log_message("日志已清空", level="info")
                
                if clear_options:
                    st.success("已清除选中的缓存")
                    st.rerun()
        
        with col2:
            if st.button("🔄 全部重置", key="cache_clear_all", type="primary", use_container_width=True):
                # 重置所有会话状态到初始值，而不是删除
                # 数据加载相关
                st.session_state.file_contents = {}
                st.session_state.file_names = []
                st.session_state.raw_texts = []
                st.session_state.uploaded_files = None
                
                # 文本预处理相关
                st.session_state.texts = None
                st.session_state.dictionary = None
                st.session_state.corpus = None
                st.session_state.stopwords = set()
                st.session_state.custom_stopwords = set()
                
                # 模型训练相关
                st.session_state.lda_model = None
                st.session_state.training_complete = False
                st.session_state.coherence_score = None
                st.session_state.perplexity = None
                st.session_state.topic_keywords = {}
                st.session_state.doc_topic_dist = None
                st.session_state.model_path = None
                st.session_state.training_time = 0
                if 'optimal_search_results' in st.session_state:
                    del st.session_state.optimal_search_results
                
                # 可视化相关
                st.session_state.pyldavis_html = None
                st.session_state.wordcloud_images = {}
                # 清除聚类缓存
                keys_to_remove = [k for k in list(st.session_state.keys()) 
                                 if k.startswith('tsne_') or k.startswith('umap_') or k.startswith('wordcloud_')]
                for key in keys_to_remove:
                    del st.session_state[key]
                
                # 系统状态
                st.session_state.current_step = None
                st.session_state.progress = 0
                st.session_state.log_messages = []
                st.session_state.start_time = time.time()  # 重置运行时间
                
                log_message("已清除所有缓存并重置系统", level="warning")
                st.success("已清除所有缓存")
                st.rerun()

def render_system_sidebar():
    """渲染系统侧边栏（只包含系统状态、日志和帮助）"""
    st.sidebar.markdown('<div style="background: linear-gradient(135deg, #2C3E50, #34495E); color: white; padding: 15px; border-radius: 10px; margin-bottom: 15px; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.2);"><h3 style="margin: 0; font-weight: 600;">🎛️ 系统控制面板</h3><p style="margin: 5px 0 0 0; font-size: 0.8rem; opacity: 0.9;">LDA 主题建模分析系统</p></div>', unsafe_allow_html=True)
    
    with st.sidebar:
        render_realtime_clock()
    
    render_system_status()
    render_load_session_dialog()
    render_cache_manager()
    render_log_area()
    render_help_section()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div style="text-align: center; padding: 10px 0; color: #7F8C8D; font-size: 0.75rem;"><p style="margin: 0;">📊 文件可视化分析系统</p><p style="margin: 3px 0;">© 2026 版权所有</p><p style="margin: 3px 0;">基于 Streamlit + Gensim</p></div>', unsafe_allow_html=True)

# 已移除废弃的render_sidebar函数
# 请使用render_system_sidebar()代替 