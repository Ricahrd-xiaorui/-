# -*- coding: utf-8 -*-
"""
字体配置工具 - 统一管理matplotlib和plotly的中文字体设置
"""

import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 全局变量缓存字体检测结果
_font_config_cache = None


def get_font_config():
    """
    获取字体配置信息
    
    返回:
        dict: 包含以下键:
            - font_found: str or None, 找到的中文字体名称
            - use_chinese: bool, 是否使用中文标签
            - matplotlib_font: list, matplotlib字体列表
            - plotly_font: str, plotly字体字符串
    """
    global _font_config_cache
    
    if _font_config_cache is not None:
        return _font_config_cache
    
    system = platform.system()
    
    # 根据操作系统定义候选中文字体
    if system == 'Windows':
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
    elif system == 'Darwin':  # macOS
        chinese_fonts = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS', 'Songti SC']
    else:  # Linux
        chinese_fonts = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 
            'Noto Sans CJK SC', 'Noto Sans SC', 
            'Source Han Sans SC', 'Source Han Sans CN',
            'Droid Sans Fallback', 'AR PL UMing CN', 'AR PL UKai CN',
            'Noto Serif CJK SC', 'DejaVu Sans'
        ]
    
    # 查找系统中可用的字体
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    font_found = None
    
    for font in chinese_fonts:
        if font in available_fonts:
            font_found = font
            break
    
    # 构建配置
    if font_found:
        matplotlib_font = [font_found, 'DejaVu Sans', 'Arial', 'sans-serif']
        plotly_font = f"{font_found}, Arial, sans-serif"
        use_chinese = True
    else:
        matplotlib_font = ['DejaVu Sans', 'Arial', 'sans-serif']
        plotly_font = "Arial, sans-serif"
        use_chinese = False
    
    _font_config_cache = {
        'font_found': font_found,
        'use_chinese': use_chinese,
        'matplotlib_font': matplotlib_font,
        'plotly_font': plotly_font,
        'system': system
    }
    
    return _font_config_cache


def setup_matplotlib_chinese():
    """
    配置matplotlib支持中文显示
    应在绑图前调用
    """
    config = get_font_config()
    plt.rcParams['font.sans-serif'] = config['matplotlib_font']
    plt.rcParams['axes.unicode_minus'] = False
    return config['use_chinese']


def get_plotly_font():
    """
    获取plotly使用的字体字符串
    
    返回:
        str: 字体字符串，如 "SimHei, Arial, sans-serif"
    """
    config = get_font_config()
    return config['plotly_font']


def get_label(chinese_text, english_text):
    """
    根据字体可用性返回中文或英文标签
    
    参数:
        chinese_text: str, 中文文本
        english_text: str, 英文文本
    
    返回:
        str: 如果有中文字体返回中文，否则返回英文
    """
    config = get_font_config()
    return chinese_text if config['use_chinese'] else english_text


# 常用标签的中英文映射
LABELS = {
    # 通用
    'topic_count': ('主题数量', 'Number of Topics'),
    'document': ('文档', 'Document'),
    'keyword': ('关键词', 'Keywords'),
    'frequency': ('频率', 'Frequency'),
    'weight': ('权重', 'Weight'),
    'score': ('分数', 'Score'),
    'optimal': ('最优', 'Optimal'),
    
    # 主题模型
    'coherence': ('连贯性分数', 'Coherence Score'),
    'coherence_title': ('主题连贯性评分 (u_mass)', 'Topic Coherence (u_mass)'),
    'coherence_ylabel': ('连贯性分数 (越接近0越好)', 'Coherence Score (closer to 0 is better)'),
    'perplexity': ('困惑度', 'Perplexity'),
    'perplexity_title': ('模型困惑度 (log值)', 'Model Perplexity (log)'),
    'perplexity_ylabel': ('困惑度 (log值，越接近0越好)', 'Perplexity (log, closer to 0 is better)'),
    
    # 词频分析
    'word': ('词语', 'Word'),
    'word_frequency': ('词频', 'Word Frequency'),
    'cooccurrence': ('共现频率', 'Co-occurrence Frequency'),
    
    # 聚类
    'cluster': ('聚类', 'Cluster'),
    'cluster_id': ('聚类ID', 'Cluster ID'),
    
    # 时序
    'time_period': ('时间段', 'Time Period'),
    'trend': ('趋势', 'Trend'),
    
    # 网络
    'node': ('节点', 'Node'),
    'edge': ('边', 'Edge'),
    'centrality': ('中心性', 'Centrality'),
    'community': ('社区', 'Community'),
}


def L(key):
    """
    获取标签的快捷函数
    
    参数:
        key: str, 标签键名
    
    返回:
        str: 对应的中文或英文标签
    """
    if key in LABELS:
        chinese, english = LABELS[key]
        return get_label(chinese, english)
    return key
