import streamlit as st
import os
import time
from datetime import datetime
from utils.session_state import save_session_state, load_session_state, log_message

def render_system_status():
    """渲染系统状态区域"""
    with st.sidebar.expander("系统状态", expanded=True):
        # 显示当前步骤和进度
        if st.session_state.get("current_step"):
            st.info(f"当前步骤: {st.session_state['current_step']}")
            st.progress(st.session_state["progress"])
        
        # 显示数据统计
        doc_count = len(st.session_state.get("raw_texts", []))
        if doc_count > 0:
            st.metric("📄 已加载文档", f"{doc_count} 个")
        
        # 会话管理按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("保存会话", key="save_session_button"):
                save_path = os.path.join("models", f"session_{int(time.time())}.json")
                save_session_state(save_path)
                st.success(f"会话已保存到: {save_path}")
                log_message(f"会话状态已保存到: {save_path}")
        
        with col2:
            if st.button("加载会话", key="load_session_button"):
                st.session_state["show_load_dialog"] = True

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
    if st.session_state.get("show_load_dialog", False):
        with st.sidebar.expander("加载会话", expanded=True):
            session_files = [f for f in os.listdir("models") if f.endswith(".json")]
            if session_files:
                selected_file = st.selectbox("选择会话文件", session_files, key="session_file_selector")
                if st.button("确认加载", key="confirm_load_session"):
                    load_path = os.path.join("models", selected_file)
                    if load_session_state(load_path):
                        st.success("会话加载成功")
                        log_message(f"会话已从 {load_path} 加载", level="success")
                        st.session_state["show_load_dialog"] = False
                        # 刷新页面以应用加载的状态
                        st.rerun()
            else:
                st.warning("未找到可用的会话文件")

# 已移除render_analysis_config()和render_visualization_options()函数
# 这些配置已移至各自的功能页面：
# - 分析配置在"模型训练"页面
# - 可视化选项在"可视化分析"页面

def render_log_area():
    """渲染日志区域"""
    with st.sidebar.expander("系统日志", expanded=False):
        # 日志显示部分
        if st.session_state.get("log_messages"):
            # 创建一个滚动区域来显示日志
            log_container = st.container()
            with log_container:
                # 显示最近的50条日志（最新的在最上面）
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
            
            # 添加日志计数器
            total_logs = len(st.session_state["log_messages"])
            if total_logs > 50:
                st.caption(f"显示最新的50条日志（共{total_logs}条）")
            else:
                st.caption(f"显示全部{total_logs}条日志记录")
        else:
            st.text("暂无日志")
        
        # 日志操作按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("清空日志", key="clear_logs_btn"):
                st.session_state["log_messages"] = []
                st.rerun()
        
        with col2:
            # 导出日志按钮
            if st.button("导出日志", key="export_logs_btn"):
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
        key="download_logs_btn"
    )
    
    log_message(f"日志已导出到: {log_path}", level="success")

def render_help_section():
    """渲染帮助和关于部分"""
    with st.sidebar.expander("帮助和关于", expanded=False):
        st.markdown("""
        ### 📊 政策文件LDA主题模型分析系统
        
        基于Streamlit开发的文本主题建模与可视化分析工具。
        
        **🔄 使用流程：**
        1. **数据加载** - 上传政策文件或使用示例数据
        2. **文本预处理** - 分词、去停用词、构建词典
        3. **模型训练** - 训练LDA主题模型
        4. **可视化分析** - 查看主题词云、分布热图等
        5. **结果导出** - 导出报告和数据
        
        **💡 小贴士：**
        - 每个功能页面都有"📖 功能介绍"可展开查看详细说明
        - 建议文档数量在3-100个之间
        - 主题数量建议3-15个
        
        **📌 版本信息：**
        - 版本: 2.0.0
        - 更新: 2026-01-12
        """)

def render_cache_manager():
    """渲染缓存管理区域"""
    with st.sidebar.expander("🗑️ 缓存管理", expanded=False):
        st.markdown("清除系统缓存以释放内存或重新开始分析。")
        
        # 显示当前缓存状态
        cache_info = []
        if st.session_state.get("raw_texts"):
            cache_info.append(f"• 原始文本: {len(st.session_state['raw_texts'])} 个")
        if st.session_state.get("texts"):
            cache_info.append(f"• 预处理文本: {len(st.session_state['texts'])} 个")
        if st.session_state.get("lda_model"):
            cache_info.append(f"• LDA模型: 已加载")
        if st.session_state.get("pyldavis_html"):
            cache_info.append(f"• PyLDAvis缓存: 已生成")
        if st.session_state.get("wordcloud_images"):
            cache_info.append(f"• 词云缓存: {len(st.session_state['wordcloud_images'])} 个")
        
        if cache_info:
            st.text("当前缓存:")
            for info in cache_info:
                st.text(info)
        else:
            st.text("当前无缓存数据")
        
        st.markdown("---")
        
        # 选择性清除
        clear_options = st.multiselect(
            "选择要清除的缓存",
            ["数据缓存", "模型缓存", "可视化缓存", "日志"],
            key="clear_cache_options"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 清除选中", key="clear_selected_cache"):
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
                    # 清除聚类缓存
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
            if st.button("🔄 全部清除", key="clear_all_cache", type="primary"):
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
    st.sidebar.header("系统控制面板")
    
    # 实时时钟放在最上面
    with st.sidebar:
        render_realtime_clock()
    
    # 渲染系统相关部分
    render_system_status()
    render_load_session_dialog()
    render_cache_manager()
    render_log_area()
    render_help_section()
    
    # 在侧边栏底部添加分隔线和版权信息
    st.sidebar.markdown("---")
    st.sidebar.caption("文件可视化分析系统 © 2026")

# 已移除废弃的render_sidebar函数
# 请使用render_system_sidebar()代替 