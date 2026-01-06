import streamlit as st
import os
import re
import jieba
import pandas as pd
import numpy as np
from collections import Counter
from gensim import corpora
from gensim.models import Phrases
from pathlib import Path
import time
from utils.session_state import get_session_state, log_message, update_progress

# 预定义的政策特定停用词
DEFAULT_POLICY_STOPWORDS = [
    "意见", "通知", "实施", "推进", "关于", "工作", "方案", "规划", "计划", "报告", 
    "决定", "部署", "要求", "办法", "细则", "规定", "条例", "安排", "部门", "地方",
    "各地", "各级", "文件", "精神", "认真", "坚持", "严格", "切实", "全面", "进一步",
    "措施", "政策", "建议", "同志", "领导", "研究", "明确", "强化", "突出", "扩大", 
    "促进", "提高", "加快", "推动", "加强", "落实"
]

# 常用中文停用词
DEFAULT_COMMON_STOPWORDS = [
    "的", "了", "和", "是", "就", "都", "而", "及", "与", "着", "或", "一个", "没有",
    "我们", "你们", "他们", "她们", "它们", "这个", "那个", "这些", "那些", "不是",
    "什么", "这样", "那样", "如此", "只是", "但是", "可是", "然而", "而且", "并且",
    "因为", "所以", "如果", "虽然", "即使", "无论", "只要", "既然", "一旦", "一直",
    "一定", "必须", "可以", "应该", "能够", "需要", "一些", "许多", "很多", "任何"
]

# 从项目根目录的stopwords.txt文件加载停用词
def load_default_stopwords(file_path="stopwords.txt"):
    """从默认文件加载停用词"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            words = f.read().strip().split('\n')
            log_message(f"已从默认文件加载 {len(words)} 个停用词")
            return set(words)
    except Exception as e:
        log_message(f"加载默认停用词文件失败: {str(e)}", level="warning")
        return set()

class TextPreprocessor:
    """文本预处理类"""
    
    def __init__(self):
        # 初始化停用词
        self.stopwords = set()
        
        # 根据会话状态决定是否加载默认的stopwords.txt文件
        if st.session_state.use_default_stopwords_file:
            default_stopwords = load_default_stopwords()
            if default_stopwords:
                self.stopwords.update(default_stopwords)
                log_message(f"已使用默认stopwords.txt作为停用词库，共 {len(default_stopwords)} 个词")
            else:
                # 如果默认文件加载失败，使用内置停用词
                self.stopwords.update(DEFAULT_COMMON_STOPWORDS)
                if st.session_state.remove_policy_words:
                    self.stopwords.update(DEFAULT_POLICY_STOPWORDS)
                log_message("默认停用词文件加载失败，使用内置停用词", level="warning")
        else:
            # 不使用默认停用词文件，使用内置停用词
            self.stopwords.update(DEFAULT_COMMON_STOPWORDS)
            if st.session_state.remove_policy_words:
                self.stopwords.update(DEFAULT_POLICY_STOPWORDS)
        
        # 从会话状态加载自定义停用词
        if st.session_state.custom_stopwords:
            self.stopwords.update(st.session_state.custom_stopwords)
        
        # 设置参数
        self.min_word_length = st.session_state.min_word_length
        self.no_below = st.session_state.no_below
        self.no_above = st.session_state.no_above
        self.min_word_count = st.session_state.min_word_count
    
    def tokenize(self, text):
        """分词处理"""
        # 清理文本
        text = re.sub(r'\s+', ' ', text)  # 合并多余空白
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)  # 只保留中文、字母、数字和空白
        
        # 使用jieba分词
        tokens = jieba.lcut(text)
        
        # 过滤停用词和短词
        filtered_tokens = [
            token for token in tokens 
            if token not in self.stopwords and len(token) >= self.min_word_length
        ]
        
        return filtered_tokens
    
    def preprocess_texts(self, texts, file_names=None):
        """预处理多个文本"""
        tokenized_texts = []
        total = len(texts)
        
        # 更新进度
        update_progress(0.0, "开始文本预处理")
        
        for i, text in enumerate(texts):
            # 更新进度
            update_progress(i/total, f"预处理文本 {i+1}/{total}")
            
            # 分词
            tokens = self.tokenize(text)
            tokenized_texts.append(tokens)
            
            # 记录日志
            if file_names and i < len(file_names):
                log_message(f"已处理文件: {file_names[i]} ({len(tokens)} 个词)")
        
        # 更新进度
        update_progress(1.0, "文本预处理完成")
        
        return tokenized_texts
    
    def create_dictionary_and_corpus(self, tokenized_texts):
        """创建词典和语料库"""
        # 更新进度
        update_progress(0.0, "开始创建词典和语料库")
        
        # 创建词典
        dictionary = corpora.Dictionary(tokenized_texts)
        
        # 过滤极端频率的词
        dictionary.filter_extremes(
            no_below=self.no_below,
            no_above=self.no_above,
            keep_n=100000  # 保留足够多的词
        )
        
        # 应用最小词频过滤
        if self.min_word_count > 1:
            # 计算词频
            word_counts = {}
            for text in tokenized_texts:
                for word in text:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # 过滤低频词
            low_freq_ids = [
                dictionary.token2id[word] 
                for word in dictionary.token2id 
                if word_counts.get(word, 0) < self.min_word_count
            ]
            dictionary.filter_tokens(low_freq_ids)
            dictionary.compactify()
            log_message(f"已过滤词频低于{self.min_word_count}的词语")
        
        # 更新进度
        update_progress(0.5, "词典创建完成")
        
        # 创建语料库 (词袋模型)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        # 记录日志
        log_message(f"词典大小: {len(dictionary)}")
        log_message(f"语料库大小: {len(corpus)}")
        
        # 更新进度
        update_progress(1.0, "语料库创建完成")
        
        return dictionary, corpus

def load_stopwords_from_file(file):
    """从文件加载停用词"""
    try:
        content = file.read().decode('utf-8')
        words = content.strip().split('\n')
        return set(words)
    except Exception as e:
        st.error(f"加载停用词文件失败: {str(e)}")
        return set()

def save_stopwords_to_file(stopwords, filename="custom_stopwords.txt"):
    """保存停用词到文件"""
    filepath = os.path.join("temp", filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        for word in sorted(stopwords):
            f.write(word + '\n')
    return filepath

def render_text_processor():
    """渲染文本预处理模块"""
    st.header("文本预处理")
    
    # 功能介绍
    with st.expander("📖 功能介绍", expanded=False):
        st.markdown("""
        **文本预处理模块** 对原始文本进行分词、清洗和特征提取，为LDA建模做准备。
        
        **主要功能：**
        - 🔤 **中文分词**：使用jieba分词器对文本进行精确分词
        - 🚫 **停用词过滤**：移除无意义的常用词和政策特定词汇
        - 📊 **词频统计**：统计词语出现频率，过滤低频和高频词
        - 📚 **词典构建**：生成用于LDA建模的词典和语料库
        
        **参数说明：**
        - **最小词长度**：过滤短于指定长度的词语（建议2-3）
        - **最小文档频率**：词语至少在多少文档中出现才保留
        - **最大文档频率**：词语最多在多少比例的文档中出现（过滤过于常见的词）
        - **最小词频**：词语在整个语料库中的最小出现次数
        
        **停用词管理：**
        - 支持使用默认停用词文件（stopwords.txt）
        - 支持添加自定义停用词
        - 支持上传停用词文件
        """)
    
    # 检查是否已加载文件
    if not st.session_state.raw_texts:
        st.warning('请先在"数据加载"选项卡中加载文件')
        return
    
    # 预处理参数设置
    with st.expander("预处理参数设置", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.min_word_length = st.slider(
                "最小词长度", 
                min_value=1, 
                max_value=5, 
                value=st.session_state.min_word_length,
                help="过滤掉短于此长度的词",
                key="min_word_length_slider"
            )
            
            st.session_state.remove_policy_words = st.checkbox(
                "移除政策特定停用词", 
                value=st.session_state.remove_policy_words,
                help="移除常见政策文件中的无意义词语",
                key="remove_policy_words_checkbox"
            )
        
        with col2:
            st.session_state.no_below = st.slider(
                "最小文档频率", 
                min_value=1, 
                max_value=10, 
                value=st.session_state.no_below,
                help="词语至少在多少文档中出现",
                key="no_below_slider"
            )
            
            st.session_state.no_above = st.slider(
                "最大文档频率", 
                min_value=0.1, 
                max_value=1.0, 
                value=st.session_state.no_above,
                step=0.05,
                help="词语最多在多少比例的文档中出现",
                key="no_above_slider"
            )
            
            st.session_state.min_word_count = st.slider(
                "最小词频", 
                min_value=1, 
                max_value=10, 
                value=st.session_state.min_word_count,
                help="词语在整个语料库中的最小出现次数",
                key="min_word_count_slider"
            )
    
    # 停用词管理
    with st.expander("停用词管理", expanded=True):
        # 添加一个复选框选择是否使用默认的stopwords.txt文件
        st.session_state.use_default_stopwords_file = st.checkbox(
            "使用默认的stopwords.txt文件作为停用词", 
            value=st.session_state.use_default_stopwords_file,
            help="选中时将优先使用项目根目录下的stopwords.txt文件作为停用词",
            key="use_default_stopwords_file_checkbox"
        )
        
        # 获取默认停用词
        default_stopwords = load_default_stopwords() if st.session_state.use_default_stopwords_file else set()
        
        # 显示当前停用词统计
        current_stopwords = set()
        if default_stopwords:
            current_stopwords.update(default_stopwords)
            st.info(f"已加载默认停用词文件(stopwords.txt)，包含 {len(default_stopwords)} 个停用词")
        else:
            current_stopwords.update(DEFAULT_COMMON_STOPWORDS)
            if st.session_state.remove_policy_words:
                current_stopwords.update(DEFAULT_POLICY_STOPWORDS)
        
        # 添加自定义停用词
        current_stopwords.update(st.session_state.custom_stopwords)
        
        st.write(f"当前停用词总数量: {len(current_stopwords)}")
        
        # 停用词管理选项
        tabs = st.tabs(["添加停用词", "上传停用词文件", "查看和编辑"])
        
        # 添加停用词标签页
        with tabs[0]:
            new_stopwords = st.text_area(
                "输入停用词(每行一个)", 
                height=150,
                help="添加自定义停用词，每行输入一个词",
                key="new_stopwords_textarea"
            )
            
            # 使用不同的方式处理按钮点击，避免会话状态冲突
            add_stopwords_clicked = st.button("添加停用词", key="add_stopwords_button")
            if add_stopwords_clicked and new_stopwords:
                words = new_stopwords.strip().split('\n')
                words = [word.strip() for word in words if word.strip()]
                if words:
                    st.session_state.custom_stopwords.update(words)
                    st.success(f"已添加 {len(words)} 个停用词")
                    log_message(f"已添加 {len(words)} 个停用词", level="success")
        
        # 上传停用词文件标签页
        with tabs[1]:
            uploaded_stopwords = st.file_uploader(
                "上传停用词文件", 
                type=["txt"], 
                help="上传包含停用词的TXT文件，每行一个词",
                key="stopwords_file_uploader"
            )
            
            if uploaded_stopwords is not None:
                load_stopwords_clicked = st.button("从文件加载停用词", key="load_stopwords_button")
                if load_stopwords_clicked:
                    new_words = load_stopwords_from_file(uploaded_stopwords)
                    if new_words:
                        st.session_state.custom_stopwords.update(new_words)
                        st.success(f"已从文件加载 {len(new_words)} 个停用词")
                        log_message(f"已从文件加载 {len(new_words)} 个停用词", level="success")
        
        # 查看和编辑标签页
        with tabs[2]:
            if current_stopwords:
                # 将停用词转换为DataFrame以便查看
                stopwords_df = pd.DataFrame({
                    "停用词": sorted(current_stopwords)
                })
                
                st.dataframe(stopwords_df, use_container_width=True, height=300)
                
                # 保存停用词
                save_stopwords_clicked = st.button("保存停用词到文件", key="save_stopwords_button")
                if save_stopwords_clicked:
                    filepath = save_stopwords_to_file(current_stopwords)
                    st.success(f"停用词已保存到: {filepath}")
                    log_message(f"停用词已保存到: {filepath}", level="success")
                
                # 清空自定义停用词
                clear_stopwords_clicked = st.button("清空自定义停用词", key="clear_custom_stopwords_button")
                if clear_stopwords_clicked:
                    st.session_state.custom_stopwords = set()
                    st.success("已清空自定义停用词")
                    log_message("已清空自定义停用词", level="success")
    
    # 开始预处理按钮
    if st.button("开始文本预处理", key="start_preprocessing_button"):
        with st.spinner("正在进行文本预处理..."):
            start_time = time.time()
            
            # 创建预处理器实例
            preprocessor = TextPreprocessor()
            
            # 处理文本
            tokenized_texts = preprocessor.preprocess_texts(
                st.session_state.raw_texts, 
                st.session_state.file_names
            )
            
            # 创建词典和语料库
            dictionary, corpus = preprocessor.create_dictionary_and_corpus(tokenized_texts)
            
            # 保存到会话状态
            st.session_state.texts = tokenized_texts
            st.session_state.dictionary = dictionary
            st.session_state.corpus = corpus
            
            # 计算并显示耗时
            elapsed_time = time.time() - start_time
            log_message(f"预处理完成，耗时: {elapsed_time:.2f}秒", level="success")
            
            # 显示成功消息
            st.success(f"文本预处理完成，耗时: {elapsed_time:.2f}秒")
    
    # 如果已经完成预处理，显示结果
    if st.session_state.texts and st.session_state.dictionary and st.session_state.corpus:
        st.subheader("预处理结果")
        
        # 显示基本统计信息
        col1, col2, col3 = st.columns(3)
        col1.metric("文档数量", len(st.session_state.texts))
        col2.metric("词典大小", len(st.session_state.dictionary))
        col3.metric("平均文档长度", f"{sum(len(text) for text in st.session_state.texts) / len(st.session_state.texts):.1f}词")
        
        # 词频统计
        with st.expander("词频统计", expanded=False):
            # 计算词频
            word_counts = Counter()
            for text in st.session_state.texts:
                word_counts.update(text)
            
            # 获取前50个高频词
            top_words = word_counts.most_common(50)
            
            # 转换为DataFrame
            df_word_counts = pd.DataFrame(top_words, columns=["词语", "频次"])
            
            # 显示词频表格
            st.dataframe(df_word_counts, use_container_width=True)
            
            # 词频直方图
            if len(top_words) > 0:
                st.bar_chart(df_word_counts.set_index("词语"))
        
        # 文档长度分布
        with st.expander("文档长度分布", expanded=False):
            doc_lengths = [len(text) for text in st.session_state.texts]
            
            # 创建DataFrame
            df_lengths = pd.DataFrame({
                "文件名": st.session_state.file_names[:len(doc_lengths)],
                "词语数量": doc_lengths
            })
            
            # 显示文档长度表格
            st.dataframe(df_lengths.sort_values("词语数量", ascending=False), use_container_width=True)
            
            # 文档长度直方图
            st.bar_chart(df_lengths.set_index("文件名"))
        
        # 预处理文本预览
        with st.expander("预处理文本预览", expanded=False):
            preview_idx = st.selectbox(
                "选择文档预览", 
                range(len(st.session_state.texts)),
                format_func=lambda i: st.session_state.file_names[i] if i < len(st.session_state.file_names) else f"文档 {i+1}",
                key="text_preview_select"
            )
            
            if preview_idx is not None:
                st.write(f"**分词结果** (共 {len(st.session_state.texts[preview_idx])} 个词):")
                st.write(" ".join(st.session_state.texts[preview_idx])) 