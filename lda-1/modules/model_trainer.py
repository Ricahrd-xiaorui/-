import streamlit as st
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from gensim.models import LdaModel, CoherenceModel
from datetime import datetime
import pickle
from pathlib import Path
from utils.session_state import get_session_state, log_message, update_progress

class LDAAnalyzer:
    """LDA主题模型分析类"""
    
    def __init__(self, texts, dictionary, corpus):
        self.texts = texts
        self.dictionary = dictionary
        self.corpus = corpus
        self.model = None
        self.coherence_score = None
        self.perplexity = None
        self.topic_keywords = {}
        self.doc_topic_dist = None
    
    def train_model(self, num_topics=5, iterations=50, passes=10, chunksize=None, 
                    alpha='auto', eta='auto', eval_every=10, random_state=42,
                    minimum_probability=0.01, decay=0.5, offset=1.0, 
                    gamma_threshold=0.001, callbacks=None):
        """
        训练LDA模型
        
        参数说明：
        ---------
        num_topics : int
            主题数量
        iterations : int
            Gibbs采样迭代次数
        passes : int
            语料库遍历次数
        chunksize : int
            在线学习批量大小
        alpha : str or list
            文档-主题分布的Dirichlet先验 ('auto', 'symmetric', 'asymmetric')
        eta : str or list
            主题-词语分布的Dirichlet先验 ('auto', 'symmetric')
        eval_every : int
            困惑度评估间隔
        random_state : int
            随机种子，确保结果可重现
        minimum_probability : float
            主题概率阈值
        decay : float
            学习率衰减参数 (0.5-1.0)
        offset : float
            学习率偏移参数
        gamma_threshold : float
            E步收敛阈值
        callbacks : dict
            回调函数字典
        """
        # 确定合适的chunksize
        if chunksize is None:
            chunksize = max(len(self.corpus) // 10, 100)
        
        # 设置随机种子以确保结果可重现
        np.random.seed(random_state)
        
        # 进度条和状态文本
        progress_bar = None
        status_text = None
        if callbacks:
            progress_bar = callbacks.get('progress_bar')
            status_text = callbacks.get('status_text')
        
        # 训练LDA模型
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            iterations=iterations,
            passes=passes,
            chunksize=chunksize,
            alpha=alpha,
            eta=eta,
            eval_every=eval_every,
            random_state=random_state,
            minimum_probability=minimum_probability,
            decay=decay,
            offset=offset,
            gamma_threshold=gamma_threshold,
            callbacks=None  # 不使用Gensim内部回调
        )
        
        # 计算困惑度
        self.perplexity = self.model.log_perplexity(self.corpus)
        
        # 计算连贯性分数
        self.coherence_score = self._calculate_coherence()
        
        # 提取主题关键词
        self.topic_keywords = {i: [word for word, prob in self.model.show_topic(i, topn=20)]
                             for i in range(num_topics)}
        
        # 计算文档-主题分布
        self.doc_topic_dist = self._get_document_topics()
        
        return self.model
    
    def _calculate_coherence(self):
        """
        计算连贯性分数
        注意：使用u_mass方法计算的连贯性分数通常为负值，越接近0表示主题一致性越好
        """
        try:
            # 提取主题关键词
            topics = []
            for i in range(self.model.num_topics):
                top_words = [word for word, _ in self.model.show_topic(i, topn=10)]
                topics.append(top_words)
            
            # 使用u_mass连贯性测量（比c_v更稳定，不需要原始文本）
            coherence_model = CoherenceModel(
                topics=topics,
                corpus=self.corpus,
                dictionary=self.dictionary,
                coherence='u_mass'  # 使用更稳定的u_mass方法
            )
            return coherence_model.get_coherence()
        except Exception as e:
            log_message(f"计算连贯性分数失败: {str(e)}", level="error")
            return None
    
    def _get_document_topics(self):
        """获取所有文档的主题分布"""
        doc_topics = []
        for i, doc in enumerate(self.corpus):
            # 获取文档的主题分布
            topics = self.model.get_document_topics(doc, minimum_probability=0.0)
            doc_topics.append([prob for _, prob in sorted(topics)])
        
        return np.array(doc_topics)
    
    def find_optimal_topics(self, start=2, end=15, step=1, random_state=42, callbacks=None):
        """
        寻找最优主题数量
        
        算法说明：
        ---------
        通过遍历不同的主题数量，训练多个LDA模型，并计算每个模型的连贯性分数和困惑度，
        最终选择连贯性最优的主题数量。
        
        注意事项：
        ---------
        1. 对于u_mass一致性测量，值通常为负，越接近0越好
        2. 为确保结果可重现，每次训练都使用固定的随机种子
        3. 搜索过程使用较少的迭代次数(iterations=50, passes=5)以加快速度
        
        参数：
        -----
        start : int
            起始主题数量
        end : int
            结束主题数量
        step : int
            步长
        random_state : int
            随机种子，确保结果可重现
        callbacks : dict
            包含进度条和状态文本的回调字典
            
        返回：
        -----
        dict : 包含最优主题数、连贯性值列表、困惑度值列表、主题范围和模型列表
        """
        coherence_values = []
        perplexity_values = []
        model_list = []
        topics_range = range(start, end+1, step)
        
        total_iterations = len(topics_range)
        
        # 进度条和状态文本
        progress_bar = None
        status_text = None
        if callbacks:
            progress_bar = callbacks.get('progress_bar')
            status_text = callbacks.get('status_text')
        
        for idx, num_topics in enumerate(topics_range):
            # 更新进度
            if progress_bar:
                progress = (idx + 1) / total_iterations
                progress_bar.progress(progress)
            
            # 更新状态
            if status_text:
                status_text.text(f"训练模型 {num_topics} 主题 ({idx+1}/{total_iterations})")
            
            # 设置随机种子以确保结果可重现
            np.random.seed(random_state)
            
            # 训练模型（搜索时使用较少的迭代以加快速度）
            model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                iterations=50,  # 搜索时使用较少的迭代次数
                passes=5,       # 搜索时使用较少的passes
                alpha='auto',
                eta='auto',
                random_state=random_state,
                callbacks=None
            )
            
            # 添加到模型列表
            model_list.append(model)
            
            # 计算困惑度
            perplexity = model.log_perplexity(self.corpus)
            perplexity_values.append(perplexity)
            
            # 计算连贯性
            try:
                # 提取主题关键词
                topics = []
                for topic_idx in range(num_topics):
                    top_words = [word for word, _ in model.show_topic(topic_idx, topn=10)]
                    topics.append(top_words)
                
                # 使用u_mass连贯性测量
                coherence_model = CoherenceModel(
                    topics=topics,
                    corpus=self.corpus,
                    dictionary=self.dictionary, 
                    coherence='u_mass'
                )
                coherence = coherence_model.get_coherence()
                coherence_values.append(coherence)
            except Exception as e:
                log_message(f"计算主题数量={num_topics}的连贯性失败: {str(e)}", level="error")
                coherence_values.append(0)
            
            # 记录日志
            log_message(f"主题数量={num_topics}, 连贯性={coherence_values[-1]:.4f}, 困惑度={perplexity_values[-1]:.4f}")
        
        # 找到最优主题数量 (对于u_mass，越接近0越好)
        if all(cv < 0 for cv in coherence_values if cv != 0):
            # 所有值都是负值，找绝对值最小的
            optimal_idx = np.argmin([abs(cv) for cv in coherence_values if cv != 0] or [0])
        else:
            # 有正值或0，找最大值
            optimal_idx = np.argmax(coherence_values)
            
        optimal_topics = topics_range[optimal_idx]
        
        return {
            'optimal_topics': optimal_topics,
            'coherence_values': coherence_values,
            'perplexity_values': perplexity_values,
            'topics_range': list(topics_range),
            'model_list': model_list
        }
    
    def save_model(self, filepath):
        """保存模型到文件"""
        if self.model:
            # 保存LDA模型
            self.model.save(filepath + ".gensim")
            
            # 保存分析器状态
            analyzer_state = {
                'coherence_score': self.coherence_score,
                'perplexity': self.perplexity,
                'topic_keywords': self.topic_keywords,
                'doc_topic_dist': self.doc_topic_dist.tolist() if self.doc_topic_dist is not None else None
            }
            
            with open(filepath + ".pkl", 'wb') as f:
                pickle.dump(analyzer_state, f)
            
            return True
        return False
    
    @classmethod
    def load_model(cls, filepath, texts, dictionary, corpus):
        """从文件加载模型"""
        try:
            # 创建分析器实例
            analyzer = cls(texts, dictionary, corpus)
            
            # 加载LDA模型
            analyzer.model = LdaModel.load(filepath + ".gensim")
            
            # 加载分析器状态
            with open(filepath + ".pkl", 'rb') as f:
                analyzer_state = pickle.load(f)
            
            analyzer.coherence_score = analyzer_state.get('coherence_score')
            analyzer.perplexity = analyzer_state.get('perplexity')
            analyzer.topic_keywords = analyzer_state.get('topic_keywords', {})
            
            doc_topic_dist = analyzer_state.get('doc_topic_dist')
            if doc_topic_dist:
                analyzer.doc_topic_dist = np.array(doc_topic_dist)
            
            return analyzer
        except Exception as e:
            log_message(f"加载模型失败: {str(e)}", level="error")
            return None

def plot_coherence_perplexity(results):
    """
    绘制连贯性和困惑度图表
    注意：
    - 对于u_mass连贯性测量，值通常为负，越接近0越好
    - 困惑度(log值)通常为负，值越大(越接近0)表示模型越好
    """
    # 设置中文字体 - 兼容不同操作系统
    import matplotlib.font_manager as fm
    import platform
    
    # 尝试查找可用的中文字体
    chinese_fonts = []
    system = platform.system()
    
    if system == 'Windows':
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    elif system == 'Darwin':  # macOS
        chinese_fonts = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 
                        'Noto Sans SC', 'Source Han Sans SC', 'Droid Sans Fallback',
                        'AR PL UMing CN', 'AR PL UKai CN']
    
    # 查找系统中可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_found = None
    for font in chinese_fonts:
        if font in available_fonts:
            font_found = font
            break
    
    if font_found:
        plt.rcParams['font.sans-serif'] = [font_found] + list(plt.rcParams['font.sans-serif'])
    else:
        # 如果没有找到中文字体，使用英文标签
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    fig = plt.figure(figsize=(12, 5))
    
    # 创建两个子图
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    # 根据是否有中文字体决定标签语言
    if font_found:
        title1 = '主题连贯性评分 (u_mass)'
        xlabel = '主题数量'
        ylabel1 = '连贯性分数 (越接近0越好)'
        title2 = '模型困惑度 (log值)'
        ylabel2 = '困惑度 (log值，越接近0越好)'
        optimal_label = '最优'
    else:
        title1 = 'Topic Coherence (u_mass)'
        xlabel = 'Number of Topics'
        ylabel1 = 'Coherence Score (closer to 0 is better)'
        title2 = 'Model Perplexity (log)'
        ylabel2 = 'Perplexity (log, closer to 0 is better)'
        optimal_label = 'Optimal'
    
    # 绘制连贯性得分
    ax1.plot(results['topics_range'], results['coherence_values'], marker='o')
    ax1.set_title(title1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1)
    ax1.grid(True, alpha=0.3)
    
    # 在最优点添加标记
    if all(cv < 0 for cv in results['coherence_values'] if cv != 0):
        # 所有值都是负值，找绝对值最小的
        abs_values = [abs(cv) for cv in results['coherence_values']]
        optimal_idx = np.argmin(abs_values)
    else:
        optimal_idx = np.argmax(results['coherence_values'])
        
    optimal_topics = results['topics_range'][optimal_idx]
    optimal_coherence = results['coherence_values'][optimal_idx]
    ax1.scatter(optimal_topics, optimal_coherence, color='red', s=100, zorder=5)
    ax1.annotate(f'{optimal_label}: {optimal_topics}',
                xy=(optimal_topics, optimal_coherence),
                xytext=(optimal_topics+1, optimal_coherence),
                arrowprops=dict(arrowstyle='->'))
    
    # 绘制困惑度
    ax2.plot(results['topics_range'], results['perplexity_values'], marker='o', color='orange')
    ax2.set_title(title2)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel2)
    ax2.grid(True, alpha=0.3)
    
    # 在困惑度最优点添加标记 (对于log困惑度，值越大越好，因为通常为负值)
    perp_idx = np.argmax(results['perplexity_values'])
    perp_topics = results['topics_range'][perp_idx]
    perp_value = results['perplexity_values'][perp_idx]
    ax2.scatter(perp_topics, perp_value, color='red', s=100, zorder=5)
    ax2.annotate(f'{optimal_label}: {perp_topics}',
                xy=(perp_topics, perp_value),
                xytext=(perp_topics+1, perp_value),
                arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    
    return fig

def render_model_trainer():
    """渲染模型训练模块"""
    st.header("LDA主题模型训练")
    
    # 功能介绍和操作手册
    with st.expander("📖 功能介绍与操作手册", expanded=False):
        st.markdown("""
        ## LDA主题模型训练模块
        
        本模块使用**LDA（Latent Dirichlet Allocation，潜在狄利克雷分配）**算法对文本进行主题建模，
        是文本挖掘和自然语言处理领域最经典的主题模型之一。
        
        ---
        
        ### 📚 算法原理
        
        LDA是一种**生成式概率模型**，假设文档由多个主题混合而成，每个主题由多个词语组成：
        
        1. **文档-主题分布 (θ)**：每个文档包含多个主题的概率分布
        2. **主题-词语分布 (φ)**：每个主题包含多个词语的概率分布
        3. **生成过程**：
           - 对每个文档，从Dirichlet分布采样主题分布 θ ~ Dir(α)
           - 对每个主题，从Dirichlet分布采样词语分布 φ ~ Dir(β)
           - 对文档中的每个词，先采样主题 z ~ Multinomial(θ)，再采样词语 w ~ Multinomial(φ_z)
        
        ---
        
        ### 🎯 主要功能
        
        | 功能 | 说明 |
        |------|------|
        | **模型训练** | 根据设定参数训练LDA主题模型 |
        | **最优主题数搜索** | 自动遍历不同主题数，找到最佳值 |
        | **模型保存/加载** | 保存训练好的模型，支持后续复用 |
        
        ---
        
        ### ⚙️ 参数详解
        
        #### 基础参数
        
        | 参数 | 说明 | 建议值 | 学术研究建议 |
        |------|------|--------|--------------|
        | **主题数量 (K)** | 模型要识别的主题个数 | 3-15 | 先用自动搜索确定，或根据领域知识设定 |
        | **迭代次数 (iterations)** | Gibbs采样的迭代轮数 | 50-200 | 学术研究建议≥100，确保收敛 |
        | **passes** | 整个语料库的遍历次数 | 5-20 | 学术研究建议≥10，小语料库可增加 |
        
        #### 高级参数
        
        | 参数 | 说明 | 选项 | 学术研究建议 |
        |------|------|------|--------------|
        | **Alpha (α)** | 文档-主题分布的Dirichlet先验 | auto/symmetric/asymmetric | **auto**：自动学习最优值（推荐）<br>**symmetric**：对称先验，所有主题权重相同<br>**asymmetric**：非对称先验，允许某些主题更常见 |
        | **Eta (β)** | 主题-词语分布的Dirichlet先验 | auto/symmetric | **auto**：自动学习最优值（推荐）<br>**symmetric**：对称先验 |
        | **Chunksize** | 在线学习的批量大小 | 100-5000 | 大语料库用2000，小语料库可用全部文档数 |
        | **随机种子** | 确保结果可重现 | 固定值 | 学术研究**必须**固定，本系统默认42 |
        | **eval_every** | 每隔多少次迭代评估困惑度 | 10-50 | 设为10可监控收敛，设为None加快训练 |
        | **minimum_probability** | 主题概率阈值 | 0.0-0.1 | 0.0保留所有主题，0.01过滤噪声 |
        
        ---
        
        ### 📊 评估指标
        
        | 指标 | 说明 | 解读 |
        |------|------|------|
        | **连贯性 (Coherence)** | 衡量主题内词语的语义一致性 | u_mass方法：值为负，**越接近0越好** |
        | **困惑度 (Perplexity)** | 衡量模型对新文档的预测能力 | log值为负，**越接近0越好** |
        
        **注意**：连贯性和困惑度可能指向不同的最优主题数，建议：
        - 优先参考**连贯性分数**（更符合人类对主题的理解）
        - 结合**领域知识**和**主题可解释性**综合判断
        
        ---
        
        ### 📋 操作流程
        
        #### 方式一：自动寻找最优主题数（推荐新手）
        
        1. 展开"寻找最优主题数量"面板
        2. 设置搜索范围（建议：起始2，结束15，步长1）
        3. 点击"寻找最优主题数量"按钮
        4. 等待搜索完成，查看连贯性和困惑度曲线
        5. 点击"直接使用最优模型"应用结果
        
        #### 方式二：手动设置参数训练
        
        1. 在"模型参数配置"中设置主题数量、迭代次数、passes
        2. 如需调整高级参数，勾选"显示高级选项"
        3. 点击"开始训练LDA模型"按钮
        4. 等待训练完成，查看结果
        
        #### 方式三：加载已有模型
        
        1. 展开"加载已有模型"面板
        2. 选择之前保存的模型文件
        3. 点击"加载模型"按钮
        
        ---
        
        ### 💡 学术研究建议
        
        1. **可重复性**：本系统已固定随机种子(42)，确保相同数据得到相同结果
        2. **参数报告**：论文中应报告主题数、迭代次数、passes、alpha、eta等参数
        3. **模型选择**：建议尝试多个主题数，结合连贯性分数和主题可解释性选择
        4. **收敛检验**：确保迭代次数足够，困惑度趋于稳定
        5. **敏感性分析**：可尝试不同参数组合，检验结果稳健性
        
        ---
        
        ### ⚠️ 常见问题
        
        | 问题 | 可能原因 | 解决方案 |
        |------|----------|----------|
        | 主题词重复度高 | 主题数过多或语料库太小 | 减少主题数或增加语料 |
        | 主题不可解释 | 预处理不充分或参数不当 | 优化停用词、调整参数 |
        | 训练时间过长 | 迭代次数过多或语料库太大 | 减少iterations/passes或增加chunksize |
        | 结果不稳定 | 随机种子未固定 | 本系统已固定，如仍不稳定请检查数据 |
        """)
    
    # 检查是否完成了预处理
    if not st.session_state.texts or not st.session_state.dictionary or not st.session_state.corpus:
        st.warning('请先在"文本预处理"选项卡中完成文本预处理')
        return
    
    # 显示语料库基本信息
    st.info(f"📊 当前语料库：{len(st.session_state.texts)} 个文档，{len(st.session_state.dictionary)} 个词汇")
    
    # 模型参数配置
    with st.expander("⚙️ 模型参数配置", expanded=True):
        st.markdown("#### 基础参数")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.num_topics = st.slider(
                "主题数量 (K)", 
                min_value=2, 
                max_value=30, 
                value=st.session_state.num_topics,
                help="LDA模型的主题数量。建议先用自动搜索确定最优值，或根据领域知识设定。",
                key="main_num_topics_slider"
            )
        
        with col2:
            st.session_state.iterations = st.number_input(
                "迭代次数 (iterations)", 
                min_value=10, 
                max_value=500, 
                value=st.session_state.iterations,
                step=10,
                help="Gibbs采样的迭代轮数。学术研究建议≥100以确保收敛。",
                key="main_iterations_input"
            )
        
        with col3:
            st.session_state.passes = st.number_input(
                "遍历次数 (passes)", 
                min_value=1, 
                max_value=50, 
                value=st.session_state.passes,
                step=1,
                help="整个语料库的遍历次数。学术研究建议≥10，小语料库可适当增加。",
                key="main_passes_input"
            )
        
        # 高级选项切换
        show_advanced = st.checkbox("🔧 显示高级选项（学术研究推荐）", value=False, key="show_advanced_options_checkbox")
        
        if show_advanced:
            st.markdown("---")
            st.markdown("#### 高级参数")
            
            # 第一行高级参数
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            
            with adv_col1:
                alpha = st.selectbox(
                    "Alpha (α) 参数", 
                    ["auto", "symmetric", "asymmetric"],
                    index=0,
                    help="""文档-主题分布的Dirichlet先验参数：
                    - auto: 自动学习最优值（推荐）
                    - symmetric: 对称先验，所有主题权重相同
                    - asymmetric: 非对称先验，允许某些主题更常见""",
                    key="model_alpha_select"
                )
            
            with adv_col2:
                eta = st.selectbox(
                    "Eta (β) 参数", 
                    ["auto", "symmetric"],
                    index=0,
                    help="""主题-词语分布的Dirichlet先验参数：
                    - auto: 自动学习最优值（推荐）
                    - symmetric: 对称先验""",
                    key="model_eta_select"
                )
            
            with adv_col3:
                # 根据语料库大小建议chunksize
                corpus_len = len(st.session_state.corpus) if st.session_state.corpus else 10
                default_chunksize = max(1, min(2000, corpus_len))
                chunksize = st.number_input(
                    "批量大小 (Chunksize)", 
                    min_value=1, 
                    max_value=10000, 
                    value=default_chunksize, 
                    step=10,
                    help="在线学习的批量大小。大语料库用2000，小语料库可用全部文档数。",
                    key="model_chunksize_input"
                )
            
            # 第二行高级参数
            adv_col4, adv_col5, adv_col6 = st.columns(3)
            
            with adv_col4:
                eval_every = st.number_input(
                    "评估间隔 (eval_every)",
                    min_value=1,
                    max_value=100,
                    value=10,
                    step=5,
                    help="每隔多少次迭代评估困惑度。设为10可监控收敛，设为较大值可加快训练。",
                    key="model_eval_every_input"
                )
            
            with adv_col5:
                minimum_probability = st.number_input(
                    "最小概率阈值",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.01,
                    step=0.01,
                    format="%.2f",
                    help="主题概率低于此阈值的将被过滤。0.0保留所有，0.01过滤噪声。",
                    key="model_min_prob_input"
                )
            
            with adv_col6:
                random_state = st.number_input(
                    "随机种子 (random_state)",
                    min_value=0,
                    max_value=9999,
                    value=42,
                    step=1,
                    help="固定随机种子确保结果可重现。学术研究必须固定此值。",
                    key="model_random_state_input"
                )
            
            # 第三行高级参数
            adv_col7, adv_col8, adv_col9 = st.columns(3)
            
            with adv_col7:
                decay = st.number_input(
                    "学习率衰减 (decay)",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    format="%.1f",
                    help="在线学习的学习率衰减参数。控制旧信息的遗忘速度。",
                    key="model_decay_input"
                )
            
            with adv_col8:
                offset = st.number_input(
                    "学习率偏移 (offset)",
                    min_value=1.0,
                    max_value=100.0,
                    value=1.0,
                    step=1.0,
                    format="%.1f",
                    help="在线学习的偏移参数。较大值使早期迭代学习率更低。",
                    key="model_offset_input"
                )
            
            with adv_col9:
                gamma_threshold = st.number_input(
                    "收敛阈值 (gamma_threshold)",
                    min_value=0.0001,
                    max_value=0.01,
                    value=0.001,
                    step=0.0001,
                    format="%.4f",
                    help="E步收敛阈值。较小值更精确但更慢。",
                    key="model_gamma_threshold_input"
                )
            
            # 显示当前参数摘要
            st.markdown("---")
            st.markdown("#### 📋 当前参数摘要（可用于论文报告）")
            params_summary = f"""
            ```
            LDA模型参数配置：
            - 主题数量 (K): {st.session_state.num_topics}
            - 迭代次数 (iterations): {st.session_state.iterations}
            - 遍历次数 (passes): {st.session_state.passes}
            - Alpha: {alpha}
            - Eta: {eta}
            - Chunksize: {chunksize}
            - 随机种子: {random_state}
            - 评估间隔: {eval_every}
            - 最小概率阈值: {minimum_probability}
            - 学习率衰减: {decay}
            - 学习率偏移: {offset}
            - 收敛阈值: {gamma_threshold}
            ```
            """
            st.markdown(params_summary)
        else:
            # 使用默认高级参数
            alpha = "auto"
            eta = "auto"
            chunksize = None
            eval_every = 10
            minimum_probability = 0.01
            random_state = 42
            decay = 0.5
            offset = 1.0
            gamma_threshold = 0.001
    
    # 自动寻找最优主题数量
    with st.expander("🔍 寻找最优主题数量", expanded=False):
        st.markdown("""
        💡 **使用说明**：自动遍历不同主题数，计算连贯性分数，找到最优值。
        搜索过程使用较少的迭代次数以加快速度，找到最优值后可直接使用或重新训练。
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            start_topics = st.number_input("起始主题数", min_value=2, max_value=15, value=2, key="start_topics_input")
        
        with col2:
            end_topics = st.number_input("结束主题数", min_value=3, max_value=30, value=15, key="end_topics_input")
        
        with col3:
            step = st.number_input("步长", min_value=1, max_value=5, value=1, key="topics_step_input")
        
        with col4:
            search_random_state = st.number_input(
                "随机种子", 
                min_value=0, 
                max_value=9999, 
                value=42, 
                key="search_random_state_input",
                help="固定随机种子确保搜索结果可重现"
            )
        
        if st.button("🔍 寻找最优主题数量", key="find_optimal_topics"):
            # 检查参数有效性
            if start_topics >= end_topics:
                st.error("起始主题数必须小于结束主题数")
            else:
                with st.spinner("正在寻找最优主题数量..."):
                    start_time = time.time()
                    
                    # 创建LDA分析器
                    analyzer = LDAAnalyzer(
                        st.session_state.texts,
                        st.session_state.dictionary,
                        st.session_state.corpus
                    )
                    
                    # 进度条
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 准备回调参数
                    callbacks = {
                        'progress_bar': progress_bar,
                        'status_text': status_text
                    }
                    
                    # 寻找最优主题数量
                    results = analyzer.find_optimal_topics(
                        start=start_topics,
                        end=end_topics,
                        step=step,
                        random_state=search_random_state,
                        callbacks=callbacks
                    )
                    
                    # 确保进度达到100%
                    progress_bar.progress(1.0)
                    status_text.text("最优主题数量搜索完成")
                    
                    # 保存搜索结果到会话状态
                    st.session_state.optimal_search_results = results
                    
                    # 绘制结果图表
                    fig = plot_coherence_perplexity(results)
                    st.pyplot(fig)
                    
                    # 显示最优主题数量
                    optimal_topics = results['optimal_topics']
                    st.success(f"已找到最优主题数量: {optimal_topics}")
                    
                    # 更新会话状态中的主题数量
                    st.session_state.num_topics = optimal_topics
                    
                    # 记录日志
                    elapsed_time = time.time() - start_time
                    log_message(f"最优主题数量搜索完成，最优值: {optimal_topics}，耗时: {elapsed_time:.2f}秒", level="success")
        
        # 如果有搜索结果，显示"使用最优模型"按钮
        if st.session_state.get("optimal_search_results"):
            results = st.session_state.optimal_search_results
            optimal_idx = results['topics_range'].index(results['optimal_topics'])
            
            st.info(f"💡 已找到最优主题数: {results['optimal_topics']}，连贯性: {results['coherence_values'][optimal_idx]:.4f}")
            
            if st.button("🚀 直接使用最优模型", key="use_optimal_model", type="primary"):
                with st.spinner("正在应用最优模型..."):
                    # 获取最优模型
                    optimal_model = results['model_list'][optimal_idx]
                    
                    # 创建分析器并设置模型
                    analyzer = LDAAnalyzer(
                        st.session_state.texts,
                        st.session_state.dictionary,
                        st.session_state.corpus
                    )
                    analyzer.model = optimal_model
                    analyzer.coherence_score = results['coherence_values'][optimal_idx]
                    analyzer.perplexity = results['perplexity_values'][optimal_idx]
                    
                    # 提取主题关键词
                    analyzer.topic_keywords = {i: [word for word, prob in optimal_model.show_topic(i, topn=20)]
                                             for i in range(optimal_model.num_topics)}
                    
                    # 计算文档-主题分布
                    analyzer.doc_topic_dist = analyzer._get_document_topics()
                    
                    # 保存到会话状态
                    st.session_state.lda_model = optimal_model
                    st.session_state.num_topics = results['optimal_topics']
                    st.session_state.coherence_score = analyzer.coherence_score
                    st.session_state.perplexity = analyzer.perplexity
                    st.session_state.topic_keywords = analyzer.topic_keywords
                    st.session_state.doc_topic_dist = analyzer.doc_topic_dist
                    st.session_state.training_complete = True
                    
                    # 保存模型
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_path = os.path.join("models", f"lda_model_{results['optimal_topics']}topics_{timestamp}")
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    
                    if analyzer.save_model(model_path):
                        st.session_state.model_path = model_path
                        log_message(f"最优模型已保存到: {model_path}", level="success")
                    
                    st.success(f"已应用最优模型（{results['optimal_topics']}个主题）")
                    log_message(f"已应用最优模型，主题数: {results['optimal_topics']}", level="success")
                    
                    # 清理搜索结果
                    del st.session_state.optimal_search_results
                    st.rerun()
    
    # 训练模型按钮
    if st.button("🚀 开始训练LDA模型", key="train_lda_model", type="primary"):
        with st.spinner("正在训练LDA模型..."):
            start_time = time.time()
            
            # 进度条和状态文本
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 创建LDA分析器
            analyzer = LDAAnalyzer(
                st.session_state.texts,
                st.session_state.dictionary,
                st.session_state.corpus
            )
            
            # 准备回调参数
            callbacks = {
                'progress_bar': progress_bar,
                'status_text': status_text
            }
            
            # 训练模型（传递所有参数）
            model = analyzer.train_model(
                num_topics=st.session_state.num_topics,
                iterations=st.session_state.iterations,
                passes=st.session_state.passes,
                chunksize=chunksize if show_advanced else None,
                alpha=alpha,
                eta=eta,
                eval_every=eval_every if show_advanced else 10,
                random_state=random_state if show_advanced else 42,
                minimum_probability=minimum_probability if show_advanced else 0.01,
                decay=decay if show_advanced else 0.5,
                offset=offset if show_advanced else 1.0,
                gamma_threshold=gamma_threshold if show_advanced else 0.001,
                callbacks=callbacks
            )
            
            # 手动更新进度到100%
            progress_bar.progress(1.0)
            status_text.text("模型训练完成")
            
            # 保存到会话状态
            st.session_state.lda_model = model
            st.session_state.coherence_score = analyzer.coherence_score
            st.session_state.perplexity = analyzer.perplexity
            st.session_state.topic_keywords = analyzer.topic_keywords
            st.session_state.doc_topic_dist = analyzer.doc_topic_dist
            st.session_state.training_complete = True
            st.session_state.training_time = time.time() - start_time
            
            # 保存模型
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join("models", f"lda_model_{st.session_state.num_topics}topics_{timestamp}")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            if analyzer.save_model(model_path):
                st.session_state.model_path = model_path
                log_message(f"模型已保存到: {model_path}", level="success")
            
            # 显示成功消息
            elapsed_time = time.time() - start_time
            st.success(f"LDA模型训练完成，耗时: {elapsed_time:.2f}秒")
            log_message(f"LDA模型训练完成，主题数: {st.session_state.num_topics}，耗时: {elapsed_time:.2f}秒", level="success")
    
    # 加载已有模型
    with st.expander("📂 加载已有模型", expanded=False):
        st.markdown("💡 加载之前保存的模型，可以继续分析或对比不同参数的结果。")
        
        # 检查models目录是否存在
        if not os.path.exists("models"):
            os.makedirs("models")
        
        model_files = [f for f in os.listdir("models") if f.endswith(".gensim")]
        
        if model_files:
            selected_model = st.selectbox("选择模型文件", model_files, key="model_file_select")
            
            if st.button("📂 加载模型", key="load_model"):
                with st.spinner("正在加载模型..."):
                    model_path = os.path.join("models", selected_model[:-7])  # 去掉.gensim后缀
                    
                    # 加载模型
                    analyzer = LDAAnalyzer.load_model(
                        model_path,
                        st.session_state.texts,
                        st.session_state.dictionary,
                        st.session_state.corpus
                    )
                    
                    if analyzer and analyzer.model:
                        # 保存到会话状态
                        st.session_state.lda_model = analyzer.model
                        st.session_state.coherence_score = analyzer.coherence_score
                        st.session_state.perplexity = analyzer.perplexity
                        st.session_state.topic_keywords = analyzer.topic_keywords
                        st.session_state.doc_topic_dist = analyzer.doc_topic_dist
                        st.session_state.training_complete = True
                        st.session_state.model_path = model_path
                        
                        st.success(f"✅ 成功加载模型: {selected_model}")
                        log_message(f"已加载模型: {selected_model}", level="success")
                    else:
                        st.error("模型加载失败")
        else:
            st.info("📭 没有找到可用的模型文件。训练模型后会自动保存到 models 目录。")
    
    # 显示训练结果
    if st.session_state.training_complete and st.session_state.lda_model:
        st.subheader("模型训练结果")
        
        # 获取模型实际的主题数量（从模型本身获取，而不是session_state）
        actual_num_topics = st.session_state.lda_model.num_topics
        
        # 显示模型基本信息
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("主题数量", actual_num_topics)
        
        with col2:
            coherence = st.session_state.coherence_score
            if coherence is not None:
                st.metric("连贯性分数", f"{coherence:.4f}")
            else:
                st.metric("连贯性分数", "N/A")
        
        with col3:
            perplexity = st.session_state.perplexity
            if perplexity is not None:
                st.metric("困惑度", f"{perplexity:.4f}")
            else:
                st.metric("困惑度", "N/A")
        
        # 显示主题关键词
        st.subheader("主题关键词")
        
        # 创建选项卡显示每个主题的关键词（使用模型实际的主题数）
        topic_tabs = st.tabs([f"主题 {i+1}" for i in range(actual_num_topics)])
        
        for i, tab in enumerate(topic_tabs):
            with tab:
                # 获取主题的关键词
                if i in st.session_state.topic_keywords:
                    keywords = st.session_state.topic_keywords[i]
                    
                    # 显示关键词列表
                    st.write(f"**主题 {i+1} 的前20个关键词**")
                    
                    # 创建关键词表格
                    keywords_df = pd.DataFrame({
                        "关键词": keywords,
                        "索引": range(1, len(keywords) + 1)
                    }).set_index("索引")
                    
                    st.dataframe(keywords_df, use_container_width=True)
                else:
                    st.write("该主题没有关键词数据")
        
        # 保存主题关键词到CSV
        if st.button("保存主题关键词到CSV", key="save_keywords"):
            # 创建一个包含所有主题关键词的DataFrame
            all_keywords = {}
            max_keywords = 0
            
            for topic_id, keywords in st.session_state.topic_keywords.items():
                all_keywords[f"主题{topic_id+1}"] = keywords
                max_keywords = max(max_keywords, len(keywords))
            
            # 确保所有列的长度相同
            for topic, keywords in all_keywords.items():
                if len(keywords) < max_keywords:
                    all_keywords[topic] = keywords + [""] * (max_keywords - len(keywords))
            
            # 创建DataFrame
            df = pd.DataFrame(all_keywords)
            
            # 保存到CSV
            csv_path = os.path.join("results", f"topic_keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, encoding="utf-8-sig", index=False)
            
            st.success(f"主题关键词已保存到: {csv_path}")
            log_message(f"主题关键词已保存到: {csv_path}", level="success") 