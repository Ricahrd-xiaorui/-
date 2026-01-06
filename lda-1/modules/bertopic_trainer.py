import streamlit as st
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
from datetime import datetime
from utils.session_state import log_message, update_progress

class BERTopicTrainer:
    """BERTopic模型训练和管理类"""
    
    def __init__(self):
        """初始化BERTopic训练器"""
        self.model = None
        self.topics = None
        self.probs = None
        self.embedding_model = None
        
    def train(self, documents, n_topics="auto", min_topic_size=5, embedding_model="paraphrase-multilingual-MiniLM-L12-v2", 
             umap_args=None, verbose=True):
        """
        训练BERTopic模型
        
        参数:
            documents: 待分析的文档列表
            n_topics: 期望的主题数量，设置为"auto"自动确定
            min_topic_size: 每个主题的最小文档数
            embedding_model: 用于嵌入文档的预训练模型名称
            umap_args: UMAP降维参数
            verbose: 是否显示详细训练信息
        """
        # 更新进度
        update_progress(0.1, "加载嵌入模型...")
        
        # 设置默认的UMAP参数
        if umap_args is None:
            umap_args = {
                "n_neighbors": 15,
                "n_components": 2,  # 降维后的维度，必须小于数据集大小
                "min_dist": 0.1
            }
        
        try:
            # 载入预训练嵌入模型
            self.embedding_model = SentenceTransformer(embedding_model)
            
            # 更新进度
            update_progress(0.3, "初始化BERTopic模型...")
            
            # 创建UMAP降维模型
            umap_model = UMAP(
                n_neighbors=umap_args.get("n_neighbors", 15),
                n_components=umap_args.get("n_components", 2),  # 降低默认维度以避免维度大于样本数
                min_dist=umap_args.get("min_dist", 0.1),
                metric='cosine',
                random_state=42
            )
            
            # 创建中文分词器
            vectorizer = CountVectorizer(
                stop_words=None,  # 我们已经在预处理中处理了停用词
                min_df=5,         # 最小文档频率
                ngram_range=(1, 2)  # 支持单词和双词组合
            )
            
            # 创建BERTopic模型
            self.model = BERTopic(
                language="chinese",  # 设置为中文
                calculate_probabilities=True,  # 计算概率分布
                nr_topics=n_topics,  # 主题数量
                min_topic_size=min_topic_size,  # 最小主题尺寸
                umap_model=umap_model,  # UMAP降维模型
                vectorizer_model=vectorizer,  # 使用自定义分词器
                embedding_model=self.embedding_model,  # 使用预训练的嵌入模型
                verbose=verbose  # 是否打印详细信息
            )
            
            # 更新进度
            update_progress(0.5, "开始训练模型...")
            
            # 训练模型
            self.topics, self.probs = self.model.fit_transform(documents)
            
            # 更新进度
            update_progress(0.9, "训练完成，生成结果...")
            
            # 记录日志
            topic_info = self.model.get_topic_info()
            num_topics = len(topic_info[topic_info['Topic'] != -1])
            total_docs = len(documents)
            log_message(f"BERTopic训练完成，发现{num_topics}个主题，共{total_docs}文档", level="success")
            
            # 更新进度
            update_progress(1.0, "BERTopic模型训练完成")
            
            return self.model, self.topics, self.probs
            
        except Exception as e:
            log_message(f"BERTopic训练失败: {str(e)}", level="error")
            raise
    
    def get_topic_info(self):
        """获取主题信息"""
        if self.model is None:
            return None
        return self.model.get_topic_info()
    
    def get_topic(self, topic_id):
        """获取指定主题的关键词"""
        if self.model is None:
            return None
        return self.model.get_topic(topic_id)
    
    def get_topic_keywords(self, n_words=10):
        """获取所有主题的关键词"""
        if self.model is None:
            return None
            
        topic_info = self.model.get_topic_info()
        topics_df = pd.DataFrame()
        
        # 排除-1主题（离群点）
        for topic in topic_info[topic_info['Topic'] != -1]['Topic']:
            keywords = self.model.get_topic(topic)[:n_words]
            topic_words = [word for word, _ in keywords]
            topic_weights = [weight for _, weight in keywords]
            
            df = pd.DataFrame({
                '主题ID': [f'主题{topic}'] * len(topic_words),
                '关键词': topic_words,
                '权重': topic_weights
            })
            topics_df = pd.concat([topics_df, df], ignore_index=True)
        
        return topics_df
    
    def save_model(self, path):
        """保存模型到文件"""
        if self.model is None:
            log_message("没有训练好的模型可保存", level="error")
            return False
        
        try:
            # 确保路径存在
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 保存模型
            self.model.save(path)
            
            # 保存主题和概率分布
            if self.topics is not None and self.probs is not None:
                np.save(f"{path}_topics.npy", np.array(self.topics))
                np.save(f"{path}_probs.npy", self.probs)
            
            log_message(f"BERTopic模型已保存到: {path}", level="success")
            return True
        except Exception as e:
            log_message(f"保存BERTopic模型失败: {str(e)}", level="error")
            return False
    
    def load_model(self, path):
        """从文件加载模型"""
        try:
            # 加载模型
            self.model = BERTopic.load(path)
            
            # 尝试加载主题和概率
            try:
                if os.path.exists(f"{path}_topics.npy"):
                    self.topics = np.load(f"{path}_topics.npy").tolist()
                
                if os.path.exists(f"{path}_probs.npy"):
                    self.probs = np.load(f"{path}_probs.npy")
            except Exception:
                pass  # 如果加载主题和概率失败，忽略错误
            
            log_message(f"BERTopic模型已从 {path} 加载", level="success")
            return self.model, self.topics, self.probs
        except Exception as e:
            log_message(f"加载BERTopic模型失败: {str(e)}", level="error")
            return None, None, None 