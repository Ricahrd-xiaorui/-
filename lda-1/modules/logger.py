import logging
import os
import streamlit as st
from datetime import datetime

def setup_logger():
    """设置日志记录器"""
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    # 获取当前日期作为日志文件名
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = f'logs/app_{today}.log'
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 创建日志记录器
    logger = logging.getLogger('lda_app')
    
    # 存储到会话状态
    st.session_state.logger = logger
    
    # 初始化日志列表
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    
    return logger

def log_message(message, level="info"):
    """
    记录日志消息
    
    参数:
        message: 日志消息
        level: 日志级别 (debug, info, warning, error, critical)
    """
    # 检查日志记录器是否已设置
    if 'logger' not in st.session_state:
        setup_logger()
    
    # 获取日志记录器
    logger = st.session_state.logger
    
    # 记录日志
    if level == "debug":
        logger.debug(message)
    elif level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "critical":
        logger.critical(message)
    elif level == "success":  # 自定义级别
        logger.info(f"[SUCCESS] {message}")
    
    # 添加到日志列表
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.log_messages.append({
        'timestamp': timestamp,
        'level': level,
        'message': message
    })

def render_logs():
    """渲染日志消息"""
    st.subheader("系统日志")
    
    # 如果没有日志消息
    if not st.session_state.log_messages:
        st.info("暂无日志记录")
        return
    
    # 日志级别过滤
    log_levels = ["全部", "debug", "info", "warning", "error", "critical", "success"]
    selected_level = st.selectbox("过滤日志级别", log_levels)
    
    # 创建日志表格
    log_data = []
    for log in reversed(st.session_state.log_messages):  # 最新的日志在前面
        if selected_level == "全部" or log['level'] == selected_level:
            # 设置日志级别的颜色
            level_color = {
                "debug": "secondary",
                "info": "info",
                "warning": "warning",
                "error": "error",
                "critical": "danger",
                "success": "success"
            }.get(log['level'], "info")
            
            # 格式化日志消息
            log_data.append({
                "时间": log['timestamp'],
                "级别": f":{level_color}[{log['level'].upper()}]",
                "消息": log['message']
            })
    
    # 显示日志表格
    if log_data:
        st.dataframe(log_data, use_container_width=True, hide_index=True)
    else:
        st.info(f"没有 {selected_level} 级别的日志记录")
    
    # 清除日志按钮
    if st.button("清除日志"):
        st.session_state.log_messages = []
        st.success("日志已清除")
        st.rerun()
    
    # 导出日志按钮
    if st.download_button(
        label="导出日志",
        data="\n".join([f"{log['timestamp']} - {log['level'].upper()} - {log['message']}" for log in st.session_state.log_messages]),
        file_name=f"lda_app_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    ):
        st.success("日志导出成功") 