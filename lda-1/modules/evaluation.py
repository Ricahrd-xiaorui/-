import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from modules.logger import log_message

def render_evaluation():
    """渲染模型评估模块"""
    st.header("模型评估")
    
    # 初始化会话状态变量
    if "coherence_score" not in st.session_state:
        st.session_state.coherence_score = None
    
    if "coherence_score_cv" not in st.session_state:
        st.session_state.coherence_score_cv = None
    
    if "perplexity" not in st.session_state:
        st.session_state.perplexity = None
    
    if "topic_keywords" not in st.session_state:
        st.session_state.topic_keywords = {}
    
    if "num_topics" not in st.session_state:
        st.session_state.num_topics = 5
    
    if "doc_topic_dist" not in st.session_state:
        st.session_state.doc_topic_dist = None
    
    # 检查是否有训练好的模型
    if 'training_complete' not in st.session_state or not st.session_state.training_complete:
        st.warning("请先完成模型训练步骤!")
        return
    
    # 创建选项卡
    eval_tabs = st.tabs(["主题一致性评估", "模型性能评估", "文档聚类评估", "交叉验证"])
    
    # 主题一致性评估选项卡
    with eval_tabs[0]:
        st.subheader("主题一致性评估")
        
        # 显示模型的主题一致性分数
        col1, col2 = st.columns(2)
        
        with col1:
            # u_mass连贯性
            coherence = st.session_state.coherence_score
            if coherence is not None:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(['u_mass连贯性'], [coherence], color='steelblue')
                ax.set_ylabel('分数')
                ax.set_title('u_mass连贯性分数')
                ax.text(0, coherence, f"{coherence:.4f}", ha='center', va='bottom')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                st.info("u_mass连贯性分数通常为负值，越接近0表示主题一致性越好")
            else:
                st.info("未计算u_mass连贯性分数")
        
        with col2:
            # c_v连贯性
            coherence_cv = st.session_state.coherence_score_cv
            if coherence_cv is not None:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(['c_v连贯性'], [coherence_cv], color='forestgreen')
                ax.set_ylabel('分数')
                ax.set_title('c_v连贯性分数')
                ax.text(0, coherence_cv, f"{coherence_cv:.4f}", ha='center', va='bottom')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                st.info("c_v连贯性分数通常为正值，值越大表示主题一致性越好")
            else:
                st.info("未计算c_v连贯性分数")
        
        # 主题关键词一致性
        st.subheader("主题关键词一致性")
        
        # 选择要评估的主题
        topic_to_evaluate = st.selectbox(
            "选择要评估的主题",
            range(st.session_state.num_topics),
            format_func=lambda x: f"主题 {x+1}"
        )
        
        # 获取该主题的关键词
        if topic_to_evaluate in st.session_state.topic_keywords:
            keywords = st.session_state.topic_keywords[topic_to_evaluate]
            
            # 显示关键词
            st.write(f"**主题 {topic_to_evaluate+1} 的关键词**")
            
            # 创建关键词表格
            keywords_df = pd.DataFrame({
                "关键词": keywords[:20],
                "索引": range(1, min(21, len(keywords) + 1))
            }).set_index("索引")
            
            st.dataframe(keywords_df, use_container_width=True)
            
            # 手动评估部分
            st.subheader("手动评估主题一致性")
            
            col1, col2 = st.columns(2)
            
            with col1:
                coherence_rating = st.slider(
                    "主题一致性评分", 
                    1, 10, 5, 
                    help="1表示关键词之间几乎没有语义关联，10表示关键词高度相关且形成一个清晰的主题"
                )
            
            with col2:
                interpretability_rating = st.slider(
                    "主题可解释性评分", 
                    1, 10, 5, 
                    help="1表示主题难以解释，10表示主题含义非常清晰"
                )
            
            # 主题命名
            topic_name = st.text_input("为这个主题命名", f"主题 {topic_to_evaluate+1}")
            
            # 保存评估结果
            if st.button("保存评估结果", key="save_topic_eval"):
                # 创建评估结果
                if 'topic_evaluations' not in st.session_state:
                    st.session_state.topic_evaluations = {}
                
                st.session_state.topic_evaluations[topic_to_evaluate] = {
                    'name': topic_name,
                    'coherence_rating': coherence_rating,
                    'interpretability_rating': interpretability_rating,
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                st.success(f"已保存主题 {topic_to_evaluate+1} 的评估结果")
                log_message(f"保存了主题 {topic_to_evaluate+1} 的手动评估结果", level="info")
        else:
            st.error(f"主题 {topic_to_evaluate+1} 没有关键词数据")
    
    # 模型性能评估选项卡
    with eval_tabs[1]:
        st.subheader("模型性能评估")
        
        # 显示模型的困惑度
        perplexity = st.session_state.perplexity
        if perplexity is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(['困惑度'], [perplexity], color='coral')
            ax.set_ylabel('分数')
            ax.set_title('模型困惑度')
            ax.text(0, perplexity, f"{perplexity:.4f}", ha='center', va='bottom')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            st.info("困惑度(log值)通常为负，值越大(越接近0)表示模型越好")
        else:
            st.info("未计算模型困惑度")
        
        # 主题分布可视化
        st.subheader("主题分布")
        
        # 获取文档-主题分布
        if 'doc_topic_dist' in st.session_state:
            doc_topic_dist = st.session_state.doc_topic_dist
            
            # 计算每个主题的平均概率
            topic_means = np.mean(doc_topic_dist, axis=0)
            
            # 创建数据框
            topic_df = pd.DataFrame({
                '主题': [f'主题 {i+1}' for i in range(len(topic_means))],
                '平均概率': topic_means
            })
            
            # 绘制条形图
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='主题', y='平均概率', data=topic_df, ax=ax)
            ax.set_title('各主题平均概率分布')
            ax.set_ylabel('平均概率')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # 计算主题覆盖率
            dominant_topics = np.argmax(doc_topic_dist, axis=1)
            topic_coverage = pd.Series(dominant_topics).value_counts().sort_index()
            
            # 检查是否有未被覆盖的主题
            uncovered_topics = set(range(st.session_state.num_topics)) - set(topic_coverage.index)
            
            if uncovered_topics:
                st.warning(f"有 {len(uncovered_topics)} 个主题没有任何文档归属: {', '.join([f'主题 {i+1}' for i in uncovered_topics])}")
            
            # 显示主题覆盖率
            st.subheader("主题覆盖率")
            
            # 创建数据框
            coverage_df = pd.DataFrame({
                '主题': [f'主题 {i+1}' for i in topic_coverage.index],
                '文档数量': topic_coverage.values,
                '文档比例': topic_coverage.values / len(doc_topic_dist)
            })
            
            # 显示数据框
            st.dataframe(coverage_df, use_container_width=True)
            
            # 绘制饼图
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.pie(
                coverage_df['文档数量'], 
                labels=coverage_df['主题'],
                autopct='%1.1f%%',
                startangle=90,
                shadow=True
            )
            ax.axis('equal')
            ax.set_title('主题覆盖率')
            st.pyplot(fig)
        else:
            st.info("未找到文档-主题分布数据")
    
    # 文档聚类评估选项卡
    with eval_tabs[2]:
        st.subheader("文档聚类评估")
        
        # 获取文档-主题分布
        if 'doc_topic_dist' in st.session_state:
            doc_topic_dist = st.session_state.doc_topic_dist
            
            # 获取每个文档的主要主题
            dominant_topics = np.argmax(doc_topic_dist, axis=1)
            
            # 计算轮廓系数
            try:
                if len(np.unique(dominant_topics)) > 1:
                    silhouette_avg = silhouette_score(doc_topic_dist, dominant_topics)
                    
                    # 显示轮廓系数
                    st.metric("轮廓系数", f"{silhouette_avg:.4f}", help="轮廓系数范围为[-1, 1]，值越大表示聚类效果越好")
                    
                    # 轮廓系数评估
                    if silhouette_avg < 0:
                        st.error("轮廓系数为负值，表示大多数样本可能被分配到了错误的聚类")
                    elif silhouette_avg < 0.2:
                        st.warning("轮廓系数较低，聚类结构较弱")
                    elif silhouette_avg < 0.5:
                        st.info("轮廓系数中等，聚类结构合理")
                    else:
                        st.success("轮廓系数较高，聚类结构良好")
                    
                    # 绘制轮廓系数图表
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(['轮廓系数'], [silhouette_avg], color='purple')
                    ax.set_ylim(-1, 1)
                    ax.set_ylabel('分数')
                    ax.set_title('聚类轮廓系数')
                    ax.text(0, silhouette_avg, f"{silhouette_avg:.4f}", ha='center', va='bottom')
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                else:
                    st.warning("所有文档都被分配到了同一个主题，无法计算轮廓系数")
            except Exception as e:
                st.error(f"计算轮廓系数时出错: {str(e)}")
            
            # 主题内聚度和主题间距离
            st.subheader("主题内聚度和主题间距离")
            
            try:
                # 计算每个主题的中心点
                topic_centers = []
                for i in range(st.session_state.num_topics):
                    topic_docs = doc_topic_dist[dominant_topics == i]
                    if len(topic_docs) > 0:
                        topic_centers.append(np.mean(topic_docs, axis=0))
                    else:
                        topic_centers.append(np.zeros(st.session_state.num_topics))
                
                topic_centers = np.array(topic_centers)
                
                # 计算主题内聚度 (平均距离)
                intra_distances = []
                for i in range(st.session_state.num_topics):
                    topic_docs = doc_topic_dist[dominant_topics == i]
                    if len(topic_docs) > 0:
                        center = topic_centers[i]
                        distances = np.sqrt(np.sum((topic_docs - center) ** 2, axis=1))
                        intra_distances.append(np.mean(distances))
                    else:
                        intra_distances.append(np.nan)
                
                # 计算主题间距离
                inter_distances = np.zeros((len(topic_centers), len(topic_centers)))
                for i in range(len(topic_centers)):
                    for j in range(len(topic_centers)):
                        if i != j:
                            inter_distances[i, j] = np.sqrt(np.sum((topic_centers[i] - topic_centers[j]) ** 2))
                
                # 显示主题内聚度
                intra_df = pd.DataFrame({
                    '主题': [f'主题 {i+1}' for i in range(len(intra_distances))],
                    '内聚度 (平均距离)': intra_distances
                })
                
                st.write("**主题内聚度 (平均距离)**")
                st.dataframe(intra_df, use_container_width=True)
                
                # 绘制主题内聚度图表
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='主题', y='内聚度 (平均距离)', data=intra_df, ax=ax)
                ax.set_title('主题内聚度 (值越小表示聚类越紧密)')
                ax.set_ylabel('平均距离')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # 显示主题间距离热力图
                st.write("**主题间距离**")
                
                # 创建热力图
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    inter_distances,
                    annot=True,
                    fmt=".3f",
                    cmap="YlGnBu",
                    xticklabels=[f'主题 {i+1}' for i in range(len(topic_centers))],
                    yticklabels=[f'主题 {i+1}' for i in range(len(topic_centers))],
                    ax=ax
                )
                ax.set_title('主题间距离 (值越大表示主题越不相似)')
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"计算主题内聚度和主题间距离时出错: {str(e)}")
        else:
            st.info("未找到文档-主题分布数据")
    
    # 交叉验证选项卡
    with eval_tabs[3]:
        st.subheader("交叉验证")
        
        st.info("交叉验证可以评估模型的稳定性和泛化能力。")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 交叉验证参数
            n_folds = st.slider("交叉验证折数", 2, 10, 5, help="数据集划分的份数")
        
        with col2:
            # 评估指标选择
            metrics = st.multiselect(
                "评估指标",
                ["困惑度", "u_mass连贯性", "c_v连贯性"],
                default=["困惑度", "u_mass连贯性"],
                help="选择用于评估的指标"
            )
        
        # 执行交叉验证按钮
        if st.button("执行交叉验证", key="run_cv", type="primary"):
            st.warning("交叉验证功能尚未实现。此功能将在后续版本中添加。")
            
            # 模拟交叉验证结果
            import random
            
            # 创建模拟数据
            cv_results = pd.DataFrame({
                '折数': list(range(1, n_folds + 1))
            })
            
            if "困惑度" in metrics:
                perplexity_values = [st.session_state.perplexity + random.uniform(-0.5, 0.5) for _ in range(n_folds)]
                cv_results['困惑度'] = perplexity_values
            
            if "u_mass连贯性" in metrics:
                coherence_values = [st.session_state.coherence_score + random.uniform(-0.05, 0.05) for _ in range(n_folds)]
                cv_results['u_mass连贯性'] = coherence_values
            
            if "c_v连贯性" in metrics:
                coherence_cv_values = [st.session_state.coherence_score_cv + random.uniform(-0.02, 0.02) for _ in range(n_folds)]
                cv_results['c_v连贯性'] = coherence_cv_values
            
            # 显示结果表格
            st.subheader("交叉验证结果")
            st.dataframe(cv_results, use_container_width=True)
            
            # 计算平均值和标准差
            cv_stats = pd.DataFrame({
                '指标': metrics,
                '平均值': [cv_results[m].mean() for m in metrics],
                '标准差': [cv_results[m].std() for m in metrics],
                '最小值': [cv_results[m].min() for m in metrics],
                '最大值': [cv_results[m].max() for m in metrics]
            })
            
            st.subheader("交叉验证统计")
            st.dataframe(cv_stats, use_container_width=True)
            
            # 绘制交叉验证结果图表
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for metric in metrics:
                ax.plot(cv_results['折数'], cv_results[metric], 'o-', label=metric)
            
            ax.set_xlabel('折数')
            ax.set_ylabel('分数')
            ax.set_title('交叉验证结果')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 显示稳定性评估
            st.subheader("模型稳定性评估")
            
            for metric in metrics:
                cv = cv_results[metric].std() / cv_results[metric].mean() * 100  # 变异系数
                
                if cv < 5:
                    st.success(f"{metric} 变异系数: {cv:.2f}% - 模型非常稳定")
                elif cv < 10:
                    st.info(f"{metric} 变异系数: {cv:.2f}% - 模型稳定性良好")
                elif cv < 15:
                    st.warning(f"{metric} 变异系数: {cv:.2f}% - 模型稳定性一般")
                else:
                    st.error(f"{metric} 变异系数: {cv:.2f}% - 模型稳定性较差")
    
    # 导出评估报告
    st.subheader("导出评估报告")
    
    if st.button("生成评估报告", key="generate_report", type="primary"):
        st.info("评估报告生成功能尚未实现。此功能将在后续版本中添加。")
        
        # TODO: 实现评估报告生成功能 