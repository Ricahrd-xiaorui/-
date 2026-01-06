import streamlit as st
import json
import time
from datetime import datetime

def get_session_state():
    """获取或创建会话状态"""
    return st.session_state

def initialize_session_state():
    """初始化会话状态变量"""
    if 'initialized' not in st.session_state:
        # 数据加载相关
        st.session_state["uploaded_files"] = None
        st.session_state["file_contents"] = {}
        st.session_state["corpus"] = None
        st.session_state["dictionary"] = None
        st.session_state["texts"] = None
        st.session_state["raw_texts"] = []
        st.session_state["file_names"] = []
        
        # 文本预处理相关
        st.session_state["stopwords"] = set()
        st.session_state["custom_stopwords"] = set()
        st.session_state["use_default_stopwords_file"] = True  # 使用默认的stopwords.txt文件
        st.session_state["min_word_length"] = 2
        st.session_state["no_below"] = 5
        st.session_state["no_above"] = 0.9
        st.session_state["min_word_count"] = 2
        st.session_state["remove_policy_words"] = True
        
        # 模型训练相关
        st.session_state["lda_model"] = None
        st.session_state["num_topics"] = 5
        st.session_state["iterations"] = 50
        st.session_state["passes"] = 10
        st.session_state["coherence_score"] = None
        st.session_state["perplexity"] = None
        st.session_state["training_complete"] = False
        st.session_state["model_path"] = None
        st.session_state["training_time"] = 0
        st.session_state["topic_keywords"] = {}
        st.session_state["doc_topic_dist"] = None
        
        # 可视化相关
        st.session_state["pyldavis_html"] = None
        st.session_state["wordcloud_images"] = {}
        st.session_state["topic_word_dist"] = None
        st.session_state["tsne_df"] = None
        st.session_state["umap_df"] = None
        st.session_state["topic_network"] = None
        st.session_state["viz_options"] = {
            'topic_wordcloud': True,
            'doc_topic_dist': True,
            'pyldavis': True,
            'doc_clustering': True,
            'topic_word_dist': True,
            'topic_similarity': True
        }
        
        # 导出相关
        st.session_state["export_formats"] = ['CSV', 'HTML', 'PNG', 'PDF']
        
        # 系统状态
        st.session_state["processing_status"] = {}
        st.session_state["current_step"] = None
        st.session_state["progress"] = 0
        st.session_state["start_time"] = time.time()
        st.session_state["log_messages"] = []
        
        # 标记初始化完成
        st.session_state["initialized"] = True

def save_session_state(path="session_state.json"):
    """保存当前会话状态到文件"""
    savable_state = {}
    for key, value in st.session_state.items():
        # 跳过不可JSON序列化的对象
        if key not in ['lda_model', 'corpus', 'dictionary', 'uploaded_files', 
                      'wordcloud_images', 'pyldavis_html', 'tsne_df', 'umap_df']:
            try:
                # 测试是否可序列化
                if isinstance(value, set):
                    savable_state[key] = list(value)
                else:
                    json.dumps(value)
                    savable_state[key] = value
            except (TypeError, OverflowError):
                pass
    
    # 添加时间戳
    savable_state['saved_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 保存到文件
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(savable_state, f, ensure_ascii=False, indent=2)
    
    return savable_state

def load_session_state(path="session_state.json"):
    """从文件加载会话状态"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            loaded_state = json.load(f)
        
        # 更新会话状态
        for key, value in loaded_state.items():
            if key in st.session_state:
                # 恢复集合类型
                if key in ['stopwords', 'custom_stopwords'] and isinstance(value, list):
                    st.session_state[key] = set(value)
                else:
                    st.session_state[key] = value
        
        return True
    except Exception as e:
        st.error(f"加载会话状态失败: {str(e)}")
        return False

def log_message(message, level="info"):
    """记录日志消息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {"time": timestamp, "message": message, "level": level}
    
    if 'log_messages' not in st.session_state:
        st.session_state["log_messages"] = []
    
    st.session_state["log_messages"].append(log_entry)
    
    # 保持日志数量合理
    if len(st.session_state["log_messages"]) > 100:
        st.session_state["log_messages"] = st.session_state["log_messages"][-100:]

def update_progress(progress_value, step_name=None):
    """更新进度"""
    st.session_state["progress"] = progress_value
    if step_name:
        st.session_state["current_step"] = step_name 