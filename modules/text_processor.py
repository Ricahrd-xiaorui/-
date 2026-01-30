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

# 获取项目根目录路径
def get_project_root():
    """获取项目根目录的绝对路径"""
    # 当前文件在 modules/ 目录下，所以父目录的父目录是项目根目录
    return Path(__file__).parent.parent.resolve()

# 从项目根目录的stopwords.txt文件加载停用词
def load_default_stopwords(file_path=None):
    """从默认文件加载停用词
    
    Args:
        file_path: 停用词文件路径，如果为None则使用项目根目录下的stopwords.txt
    """
    try:
        if file_path is None:
            # 使用项目根目录下的stopwords.txt
            file_path = get_project_root() / "stopwords.txt"
        else:
            file_path = Path(file_path)
        
        if not file_path.exists():
            log_message(f"停用词文件不存在: {file_path}", level="warning")
            return set()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            words = f.read().strip().split('\n')
            # 过滤空行
            words = [w.strip() for w in words if w.strip()]
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
    
    # 功能介绍与操作手册
    with st.expander("📖 功能介绍与操作手册", expanded=False):
        st.markdown("""
        ## 🔧 文本预处理模块
        
        **功能概述**：对原始文本进行分词、清洗和特征提取，为LDA主题建模做准备。
        
        ---
        
        ### 🎯 使用场景
        
        | 场景 | 参数建议 | 说明 |
        |------|----------|------|
        | 政策文本分析 | 最小词长=2，移除政策停用词 | 过滤政策文件中的常见无意义词 |
        | 学术论文研究 | 最小词长=2，最小文档频率=2 | 确保词语具有统计意义 |
        | 短文本分析 | 最小词长=1，最小文档频率=1 | 保留更多词语信息 |
        | 大规模语料 | 最大文档频率=0.5，最小词频=3 | 过滤过于常见和稀有的词 |
        
        ---
        
        ### 📋 操作步骤
        
        **步骤1：配置专业词典（可选但推荐）**
        1. 展开"专业词典管理"
        2. 添加领域专业术语（如"乡村振兴"、"数字经济"等）
        3. 这些词汇会被作为整体识别，不会被错误分词
        
        **步骤2：设置预处理参数**
        1. 展开"预处理参数设置"
        2. 根据研究需要调整各项参数
        3. 参考下方参数说明表格
        
        **步骤3：配置停用词**
        1. 展开"停用词管理"
        2. 选择是否使用默认停用词文件
        3. 可添加自定义停用词或上传停用词文件
        
        **步骤4：执行预处理**
        1. 点击"开始文本预处理"按钮
        2. 等待处理完成
        3. 查看预处理结果统计
        
        ---
        
        ### ⚙️ 参数详解
        
        | 参数 | 范围 | 默认值 | 说明 | 学术研究建议 |
        |------|------|--------|------|--------------|
        | 最小词长度 | 1-5 | 2 | 过滤短于此长度的词 | 中文建议2，保留双字词 |
        | 最小文档频率 | 1-10 | 2 | 词语至少出现在多少文档中 | 建议2-3，过滤偶发词 |
        | 最大文档频率 | 0.1-1.0 | 0.85 | 词语最多出现在多少比例的文档中 | 建议0.5-0.8，过滤过于常见的词 |
        | 最小词频 | 1-10 | 2 | 词语在语料库中的最小出现次数 | 建议2-5，确保统计意义 |
        
        ---
        
        ### 🔤 分词算法说明
        
        本系统使用 **jieba分词器** 进行中文分词：
        - **精确模式**：将句子最精确地切开，适合文本分析
        - **用户词典**：支持添加专业术语，提高分词准确性
        - **词性标注**：可选功能，用于后续词性筛选
        
        ---
        
        ### 🚫 停用词说明
        
        **停用词类型：**
        - **通用停用词**：的、了、和、是、在等常见虚词
        - **政策停用词**：意见、通知、实施、推进等政策文件常见词
        - **自定义停用词**：用户根据研究需要添加的词语
        
        **停用词来源优先级：**
        1. 默认stopwords.txt文件（如启用）
        2. 内置通用停用词
        3. 政策特定停用词（如启用）
        4. 用户自定义停用词
        
        ---
        
        ### 💡 使用建议
        
        **学术研究建议：**
        - 记录所有预处理参数设置，便于论文方法部分撰写
        - 建议先用默认参数预处理，查看结果后再调整
        - 关注词频统计中的高频词，判断是否需要添加停用词
        
        **参数调优建议：**
        - 如果主题词中出现太多无意义词，增加停用词
        - 如果词典太小，降低最小文档频率和最小词频
        - 如果词典太大，提高最小文档频率或降低最大文档频率
        
        ---
        
        ### ❓ 常见问题
        
        **Q: 专业术语被错误分词怎么办？**
        A: 在"专业词典管理"中添加该术语，系统会将其作为整体识别。
        
        **Q: 预处理后词典太小怎么办？**
        A: 降低最小文档频率和最小词频参数，或减少停用词。
        
        **Q: 如何判断预处理效果好不好？**
        A: 查看词频统计中的高频词，应该是有意义的实词；查看文档长度分布，不应有过短的文档。
        
        **Q: 预处理参数如何在论文中报告？**
        A: 建议报告：分词工具（jieba）、最小词长、文档频率范围、停用词来源等。
        """)
    
    # 检查是否已加载文件
    if not st.session_state.raw_texts:
        st.warning('请先在"数据加载"选项卡中加载文件')
        return
    
    # ========== 专业词典管理（放在最前面，因为影响分词结果）==========
    with st.expander("📚 专业词典管理", expanded=False):
        st.markdown("""
        **专业词典** 用于提高分词准确性，确保专业术语被正确识别。
        词典中的词汇会被添加到jieba分词器的用户词典中。
        
        ⚠️ **注意**：请在执行预处理之前配置好词典！
        
        💡 **提示**：系统不内置专业词典，请根据您的研究领域自行准备专业术语词典。
        可从以下渠道获取：
        - 学术论文中的关键词列表
        - 行业标准术语表
        - 政策文件中的专有名词
        - 搜狗细胞词库等公开资源
        """)
        
        try:
            from modules.dictionary_manager import DictionaryManager, render_dictionary_manager_compact
            
            # 初始化词典管理器
            if st.session_state.get("dictionary_manager") is None:
                st.session_state["dictionary_manager"] = DictionaryManager()
            
            # 渲染紧凑版词典管理界面
            render_dictionary_manager_compact()
            
        except ImportError:
            st.info("📦 词典管理模块正在加载中...")
            
            # 提供简单的词典输入功能作为备选
            custom_dict_words = st.text_area(
                "输入专业词汇（每行一个）",
                height=100,
                help="这些词汇会被添加到jieba分词器中",
                key="simple_custom_dict"
            )
            
            if st.button("应用专业词汇", key="apply_simple_dict"):
                if custom_dict_words:
                    words = [w.strip() for w in custom_dict_words.strip().split('\n') if w.strip()]
                    for word in words:
                        jieba.add_word(word)
                    st.success(f"已添加 {len(words)} 个专业词汇到分词器")
                    log_message(f"已添加 {len(words)} 个专业词汇", level="success")
    
    # ========== 预处理参数设置 ==========
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