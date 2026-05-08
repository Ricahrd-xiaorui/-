import streamlit as st
import json
import time
from datetime import datetime
from io import BytesIO
import base64

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
        
        # ========== 学术研究文本分析功能扩展 ==========
        
        # 质性编码模块 (Requirements 1.1-1.7)
        st.session_state["coding_scheme"] = None  # CodingScheme实例
        st.session_state["coded_segments"] = []   # List[CodedSegment]
        
        # 词频与共现分析模块 (Requirements 2.1-2.7)
        st.session_state["word_frequency"] = {}   # Dict[str, int] 词频统计结果
        st.session_state["pos_tags"] = []         # List[List[str]] 词性标注结果
        st.session_state["cooccurrence_matrix"] = {}  # Dict[Tuple[str, str], int] 共现矩阵
        st.session_state["cooccurrence_window_size"] = 5  # 共现窗口大小
        st.session_state["cooccurrence_min_freq"] = 2     # 最小共现频率阈值
        
        # 文本聚类与分类模块 (Requirements 3.1-3.7)
        st.session_state["cluster_labels"] = None  # np.ndarray 聚类标签
        st.session_state["cluster_algorithm"] = "kmeans"  # 聚类算法: kmeans, hierarchical
        st.session_state["num_clusters"] = 3       # 聚类数量
        st.session_state["cluster_keywords"] = {}  # Dict[int, List[str]] 聚类关键词
        st.session_state["classification_labels"] = {}  # Dict[str, str] 分类标签
        st.session_state["classifier_model"] = None     # 分类器模型
        
        # 时序演变分析模块 (Requirements 4.1-4.7)
        st.session_state["time_labels"] = {}      # Dict[str, str] 文档时间标签
        st.session_state["keyword_trends"] = {}   # Dict[str, Dict[str, int]] 关键词趋势
        st.session_state["topic_evolution"] = {}  # Dict[str, List[float]] 主题演变
        
        # 文本比较分析模块 (Requirements 5.1-5.7)
        st.session_state["similarity_matrix"] = None  # np.ndarray 相似度矩阵
        st.session_state["common_keywords"] = []      # List[str] 共同关键词
        st.session_state["unique_keywords"] = {}      # Dict[str, List[str]] 差异关键词
        st.session_state["similar_segments"] = []     # List[Tuple] 相似段落
        
        # 引用与参考分析模块 (Requirements 6.1-6.7)
        st.session_state["citations"] = {}        # Dict[str, List[str]] 引用关系
        st.session_state["citation_network"] = None  # nx.DiGraph 引用网络
        st.session_state["core_documents"] = []   # List[Tuple[str, int]] 核心文档
        st.session_state["citation_analyzer"] = None  # CitationAnalyzer实例
        
        # 语义网络分析模块 (Requirements 7.1-7.8)
        st.session_state["semantic_network"] = None  # nx.Graph 语义网络
        st.session_state["community_labels"] = {}    # Dict[str, int] 社区标签
        st.session_state["centrality_metrics"] = {}  # Dict[str, Dict[str, float]] 中心性指标
        st.session_state["center_word"] = None       # str 核心概念词
        
        # 文本统计与可读性分析模块 (Requirements 8.1-8.7)
        st.session_state["text_statistics"] = {}  # Dict[str, dict] 文本统计结果
        st.session_state["readability_scores"] = {}  # Dict[str, float] 可读性指数
        
        # 专业词典管理模块 (Requirements 9.1-9.10)
        st.session_state["dictionary_manager"] = None  # DictionaryManager实例
        st.session_state["active_dictionaries"] = []   # List[str] 激活的词典名称
        st.session_state["term_frequencies"] = {}      # Dict[str, int] 术语频率统计
        
        # 标记初始化完成
        st.session_state["initialized"] = True

def register_chart(chart_id, chart_type, title, description="", data=None, figure_data=None, image_base64=None):
    """注册一个图表到图表注册表

    Args:
        chart_id: 图表唯一标识符
        chart_type: 图表类型 (plotly, matplotlib, wordcloud, network, etc.)
        title: 图表标题
        description: 图表描述
        data: 图表相关数据
        figure_data: 图表的figure数据 (plotly fig 或 matplotlib fig)
        image_base64: 图表的base64编码图像数据
    """
    if 'chart_registry' not in st.session_state:
        st.session_state['chart_registry'] = {}

    st.session_state['chart_registry'][chart_id] = {
        'chart_type': chart_type,
        'title': title,
        'description': description,
        'data': data,
        'figure_data': figure_data,
        'image_base64': image_base64,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def get_all_charts():
    """获取所有已注册的图表"""
    return st.session_state.get('chart_registry', {})

def get_chart(chart_id):
    """获取指定ID的图表"""
    registry = st.session_state.get('chart_registry', {})
    return registry.get(chart_id)

def clear_chart_registry():
    """清空图表注册表"""
    if 'chart_registry' in st.session_state:
        st.session_state['chart_registry'] = {}

def export_chart_to_base64(fig, format='png', dpi=150):
    """将matplotlib或plotly图表导出为base64

    Args:
        fig: matplotlib或plotly图表对象
        format: 图片格式 (png, jpg, svg)
        dpi: 图片分辨率

    Returns:
        str: base64编码的图片数据
    """
    try:
        buf = BytesIO()

        if hasattr(fig, 'savefig'):
            fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight', facecolor='white')
        elif hasattr(fig, 'to_image'):
            img_bytes = fig.to_image(format=format, width=1200, height=600)
            buf.write(img_bytes)
        else:
            return None

        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/{format};base64,{img_base64}"
    except Exception as e:
        print(f"图表导出失败: {e}")
        return None

def save_session_state(path="session_state.json"):
    """保存当前会话状态到文件"""
    import os
    savable_state = {}
    skip_keys = [
        'lda_model', 'corpus', 'dictionary', 'uploaded_files',
        'wordcloud_images', 'pyldavis_html', 'tsne_df', 'umap_df',
        'coding_scheme', 'classifier_model', 'citation_network',
        'semantic_network', 'dictionary_manager', 'cluster_labels',
        'similarity_matrix', 'chart_registry', 'initialized',
        'progress', 'current_step', 'log_messages', 'start_time'
    ]

    for key, value in st.session_state.items():
        if key not in skip_keys:
            try:
                if isinstance(value, set):
                    savable_state[key] = list(value)
                else:
                    json.dumps(value)
                    savable_state[key] = value
            except (TypeError, OverflowError):
                pass

    savable_state['saved_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    savable_state['saved_version'] = '1.0'

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(savable_state, f, ensure_ascii=False, indent=2)

    return savable_state

def load_session_state(path="session_state.json"):
    """从文件加载会话状态"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            loaded_state = json.load(f)

        set_keys = ['stopwords', 'custom_stopwords']

        for key, value in loaded_state.items():
            if key.startswith('saved_'):
                continue
            if key in st.session_state:
                if key in set_keys and isinstance(value, list):
                    st.session_state[key] = set(value)
                elif isinstance(value, dict) and isinstance(st.session_state[key], dict):
                    st.session_state[key].update(value)
                else:
                    st.session_state[key] = value

        return True
    except FileNotFoundError:
        st.error(f"会话文件不存在: {path}")
        return False
    except json.JSONDecodeError:
        st.error(f"会话文件格式错误: {path}")
        return False
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