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
                    alpha='auto', eta='auto', eval_every=10, callbacks=None):
        """训练LDA模型"""
        # 确定合适的chunksize
        if chunksize is None:
            chunksize = max(len(self.corpus) // 10, 100)
        
        # 设置随机种子以确保结果可重现
        np.random.seed(42)
        
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
    
    def find_optimal_topics(self, start=2, end=15, step=1, callbacks=None):
        """
        寻找最优主题数量
        注意：对于u_mass一致性测量，值通常为负，越接近0越好
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
            
            # 训练模型
            model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                iterations=50,  # 使用较少的迭代次数加快搜索
                passes=5,      # 使用较少的passes加快搜索
                alpha='auto',
                eta='auto',
                callbacks=None  # 不使用回调函数
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
                    coherence='u_mass'  # 使用与_calculate_coherence相同的方法
                )
                coherence = coherence_model.get_coherence()
                coherence_values.append(coherence)
            except Exception as e:
                log_message(f"计算主题数量={num_topics}的连贯性失败: {str(e)}", level="error")
                coherence_values.append(0)
            
            # 记录日志
            log_message(f"主题数量={num_topics}, 连贯性={coherence_values[-1]:.4f}, 困惑度={perplexity_values[-1]:.4f}")
        
        # 找到最优主题数量 (对于u_mass，寻找最大值而非最小值，因为越接近0越好)
        # 注意：如果coherence_values全为负值，则选择绝对值最小的作为最优
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
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    fig = plt.figure(figsize=(12, 5))
    
    # 创建两个子图
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    # 绘制连贯性得分
    ax1.plot(results['topics_range'], results['coherence_values'], marker='o')
    ax1.set_title('主题连贯性评分 (u_mass)')
    ax1.set_xlabel('主题数量')
    ax1.set_ylabel('连贯性分数 (越接近0越好)')
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
    ax1.annotate(f'最优: {optimal_topics}',
                xy=(optimal_topics, optimal_coherence),
                xytext=(optimal_topics+1, optimal_coherence),
                arrowprops=dict(arrowstyle='->'))
    
    # 绘制困惑度
    ax2.plot(results['topics_range'], results['perplexity_values'], marker='o', color='orange')
    ax2.set_title('模型困惑度 (log值)')
    ax2.set_xlabel('主题数量')
    ax2.set_ylabel('困惑度 (log值，越接近0越好)')
    ax2.grid(True, alpha=0.3)
    
    # 在困惑度最优点添加标记 (对于log困惑度，值越大越好，因为通常为负值)
    perp_idx = np.argmax(results['perplexity_values'])
    perp_topics = results['topics_range'][perp_idx]
    perp_value = results['perplexity_values'][perp_idx]
    ax2.scatter(perp_topics, perp_value, color='red', s=100, zorder=5)
    ax2.annotate(f'最优: {perp_topics}',
                xy=(perp_topics, perp_value),
                xytext=(perp_topics+1, perp_value),
                arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    
    return fig

def render_model_trainer():
    """渲染模型训练模块"""
    st.header("LDA主题模型训练")
    
    # 功能介绍
    with st.expander("📖 功能介绍", expanded=False):
        st.markdown("""
        **模型训练模块** 使用LDA（潜在狄利克雷分配）算法对文本进行主题建模。
        
        **主要功能：**
        - 🎯 **模型训练**：根据设定参数训练LDA主题模型
        - 🔍 **最优主题数搜索**：自动寻找最佳主题数量
        - 💾 **模型保存/加载**：保存训练好的模型，支持后续加载使用
        
        **核心参数说明：**
        - **主题数量**：模型要识别的主题个数（建议3-15个）
        - **迭代次数**：模型训练的迭代轮数（越多越精确，但耗时更长）
        - **passes**：整个语料库的遍历次数
        
        **高级参数：**
        - **Alpha**：文档-主题分布的先验参数（auto自动优化）
        - **Eta**：主题-词语分布的先验参数（auto自动优化）
        - **Chunksize**：每次训练的文档批量大小
        
        **评估指标：**
        - **连贯性分数(Coherence)**：衡量主题内词语的语义一致性，使用u_mass方法，值越接近0越好
        - **困惑度(Perplexity)**：衡量模型对新文档的预测能力，log值越接近0越好
        
        **使用建议：**
        1. 首次使用建议先用"寻找最优主题数量"功能
        2. 找到最优主题数后，可直接使用最优模型或手动调整参数重新训练
        """)
    
    # 检查是否完成了预处理
    if not st.session_state.texts or not st.session_state.dictionary or not st.session_state.corpus:
        st.warning('请先在"文本预处理"选项卡中完成文本预处理')
        return
    
    # 合并"模型参数配置"和"高级模型参数"为一个配置区域
    with st.expander("模型参数配置", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.num_topics = st.slider(
                "主题数量", 
                min_value=2, 
                max_value=20, 
                value=st.session_state.num_topics,
                help="LDA模型的主题数量",
                key="main_num_topics_slider"
            )
        
        with col2:
            st.session_state.iterations = st.number_input(
                "迭代次数", 
                min_value=10, 
                max_value=200, 
                value=st.session_state.iterations,
                step=10,
                help="LDA模型训练的迭代次数",
                key="main_iterations_input"
            )
        
        with col3:
            st.session_state.passes = st.number_input(
                "passes", 
                min_value=1, 
                max_value=20, 
                value=st.session_state.passes,
                step=1,
                help="LDA训练中的passes数量",
                key="main_passes_input"
            )
        
        # 高级选项切换
        show_advanced = st.checkbox("显示高级选项", value=False, key="show_advanced_options_checkbox")
        
        if show_advanced:
            st.markdown("---")
            st.subheader("高级参数")
            advanced_col1, advanced_col2, advanced_col3 = st.columns(3)
            
            with advanced_col1:
                alpha = st.radio(
                    "Alpha参数", 
                    ["auto", "symmetric", "asymmetric"],
                    index=0,
                    help="主题-文档分布的先验参数",
                    key="model_alpha_radio"
                )
            
            with advanced_col2:
                eta = st.radio(
                    "Eta参数", 
                    ["auto", "symmetric"],
                    index=0,
                    help="词-主题分布的先验参数",
                    key="model_eta_radio"
                )
            
            with advanced_col3:
                chunksize = st.number_input(
                    "Chunksize", 
                    min_value=100, 
                    max_value=5000, 
                    value=2000, 
                    step=100,
                    help="每次训练的文档批量大小",
                    key="model_chunksize_input"
                )
        else:
            alpha = "auto"
            eta = "auto"
            chunksize = None
    
    # 自动寻找最优主题数量
    with st.expander("寻找最优主题数量", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_topics = st.number_input("起始主题数", min_value=2, max_value=15, value=2, key="start_topics_input")
        
        with col2:
            end_topics = st.number_input("结束主题数", min_value=3, max_value=20, value=15, key="end_topics_input")
        
        with col3:
            step = st.number_input("步长", min_value=1, max_value=5, value=1, key="topics_step_input")
        
        if st.button("寻找最优主题数量", key="find_optimal_topics"):
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
                        callbacks=callbacks  # 传递回调参数
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
    if st.button("开始训练LDA模型", key="train_lda_model"):
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
            
            # 训练模型
            model = analyzer.train_model(
                num_topics=st.session_state.num_topics,
                iterations=st.session_state.iterations,
                passes=st.session_state.passes,
                chunksize=chunksize if show_advanced else None,
                alpha=alpha if show_advanced else 'auto',
                eta=eta if show_advanced else 'auto',
                callbacks=callbacks  # 传递包含UI元素的字典
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
    with st.expander("加载已有模型", expanded=False):
        model_files = [f for f in os.listdir("models") if f.endswith(".gensim")]
        
        if model_files:
            selected_model = st.selectbox("选择模型文件", model_files, key="model_file_select")
            
            if st.button("加载模型", key="load_model"):
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
                        
                        st.success(f"成功加载模型: {selected_model}")
                        log_message(f"已加载模型: {selected_model}", level="success")
                    else:
                        st.error("模型加载失败")
        else:
            st.info("没有找到可用的模型文件")
    
    # 显示训练结果
    if st.session_state.training_complete and st.session_state.lda_model:
        st.subheader("模型训练结果")
        
        # 显示模型基本信息
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("主题数量", st.session_state.num_topics)
        
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
        
        # 创建选项卡显示每个主题的关键词
        topic_tabs = st.tabs([f"主题 {i+1}" for i in range(st.session_state.num_topics)])
        
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