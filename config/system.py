# -*- coding: utf-8 -*-
"""
系统配置 - 统一管理系统路径和常量配置
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

TEMP_DIR = PROJECT_ROOT / "temp"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
FONTS_DIR = PROJECT_ROOT / "fonts"

STOPWORDS_FILE = PROJECT_ROOT / "stopwords.txt"

def ensure_directories():
    """确保所有必要的目录存在"""
    for directory in [TEMP_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, FONTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

DEFAULT_POLICY_STOPWORDS = [
    "意见", "通知", "实施", "推进", "关于", "工作", "方案", "规划", "计划", "报告",
    "决定", "部署", "要求", "办法", "细则", "规定", "条例", "安排", "部门", "地方",
    "各地", "各级", "文件", "精神", "认真", "坚持", "严格", "切实", "全面", "进一步",
    "措施", "政策", "建议", "同志", "领导", "研究", "明确", "强化", "突出", "扩大",
    "促进", "提高", "加快", "推动", "加强", "落实"
]

DEFAULT_COMMON_STOPWORDS = [
    "的", "了", "和", "是", "就", "都", "而", "及", "与", "着", "或", "一个", "没有",
    "我们", "你们", "他们", "她们", "它们", "这个", "那个", "这些", "那些", "不是",
    "什么", "这样", "那样", "如此", "只是", "但是", "可是", "然而", "而且", "并且",
    "因为", "所以", "如果", "虽然", "即使", "无论", "只要", "既然", "一旦", "一直",
    "一定", "必须", "可以", "应该", "能够", "需要", "一些", "许多", "很多", "任何"
]

MAX_FILES = 10000
BATCH_SIZE = 100
MEMORY_WARNING_THRESHOLD = 1000

SUPPORTED_FILE_EXTENSIONS = ['.txt']

DEFAULT_TEXT_PREPROCESSOR_CONFIG = {
    'min_word_length': 2,
    'no_below': 5,
    'no_above': 0.9,
    'min_word_count': 2,
    'remove_policy_words': True,
    'use_default_stopwords_file': True
}

DEFAULT_LDA_CONFIG = {
    'num_topics': 5,
    'iterations': 50,
    'passes': 10,
    'random_state': 42
}
