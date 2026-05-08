# 政策文件LDA主题模型可视化分析系统

面向学术研究的中文政策文本LDA主题建模分析平台，基于Streamlit开发。系统集成双层词典干预机制、变分推断LDA优化、词频共现分析、文本聚类、时序演变、比较分析、语义网络等多种分析功能。

## ✨ 核心特性

- 🎯 **双层词典干预**：通用政策词典 + 领域专用词典，提升分词准确性
- 🎯 **LDA主题建模**：在线变分推断 + 困惑度-连贯性双指标寻优 + 超参数自适应
- 📊 **文本统计**：字符数、词数、句数、TTR等多维度统计
- 🔢 **词频与共现**：词频统计、词语共现矩阵、共现网络图
- 🎨 **聚类分类**：K-means/层次聚类 + KNN自动分类
- 📅 **时序分析**：关键词趋势追踪、主题强度演变
- 🔍 **比较分析**：文档相似度、差异关键词识别
- 🕸️ **语义网络**：概念关联网络、社区检测、中心性分析
- 🏷️ **质性编码**：文本编码标注、编码体系管理
- � **用户管理**：MySQL数据库 + RBAC权限控制
- 💾 **结果导出**：图表PNG/PDF + 数据CSV/Excel + 综合HTML报告

## 🚀 快速开始

### 环境要求

- Python 3.9+
- MySQL 8.0+（用户认证功能需要）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 关键依赖说明

| 依赖库 | 用途 | 是否必需 |
|--------|------|----------|
| streamlit | Web应用框架 | ✅ 必需 |
| jieba | 中文分词引擎 | ✅ 必需 |
| gensim | LDA主题模型（变分推断） | ✅ 必需 |
| pyLDAvis | LDA交互式可视化 | ✅ 必需 |
| plotly | 交互式图表（热图/散点图/折线图/网络图） | ✅ 必需 |
| wordcloud | 词云图生成 | ✅ 必需 |
| scikit-learn | 聚类(K-means/层次)、分类(KNN)、TF-IDF、t-SNE | ✅ 必需 |
| networkx | 语义网络构建与分析（社区检测/中心性） | ✅ 必需 |
| pandas / numpy | 数据处理与数值计算 | ✅ 必需 |
| matplotlib | 静态图表绘制 | ✅ 必需 |
| pdfplumber | PDF文件解析 | ✅ 必需 |
| python-docx | Word(.docx)文件解析 | ⚠️ 可选（无此库则禁用docx上传） |
| pymysql | MySQL数据库驱动 | ⚠️ 可选（无此库则禁用用户登录） |

> **注意**：plotly 和 networkx 是显示网络图、热力图等可视化组件的必要依赖，请确保已安装。

### 运行应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可使用。

## 📖 使用流程

```
1. 用户登录/注册 → 2. 上传政策文件(TXT/PDF/Word/ZIP)
    ↓
3. 文本预处理(jieba分词 + 双层词典 + 停用词过滤)
    ↓
4. 基础文本分析(统计 / 词频 / 共现) ← 探索性分析
    ↓
5. LDA主题建模(自动寻优K值 + 变分推断训练)
    ↓
6. 主题可视化(词云 / PyLDAvis / 热图 / t-SNE)
    ↓
7. 高级分析(聚类 / 时序 / 比较 / 语义网络)
    ↓
8. 结果导出(图表 + 数据 + 综合报告)
```

## � 项目结构

```
d:\project\lda-1\
├── app.py                          # 主入口（页面路由/UI组装）
├── requirements.txt                # 依赖声明
├── stopwords.txt                   # 默认停用词表
├── 通用政策专业词典.txt             # 内置通用政策词典
│
├── config/
│   ├── system.py                   # 系统常量配置
│   └── database.py                 # 数据库配置
│
├── modules/                        # 功能模块（23个）
│   ├── auth.py                     # 用户认证（注册/登录/RBAC/日志）
│   ├── sidebar.py                  # 侧边栏导航
│   ├── data_loader.py              # 数据加载（PDF/Word/TXT/ZIP）
│   ├── text_processor.py           # 文本预处理（jieba+词典+停用词）
│   ├── dictionary_manager.py       # 双层词典管理（CRUD/激活/jieba集成）
│   ├── text_analysis_readability.py # 文本统计分析
│   ├── word_frequency_cooccurrence.py # 词频与共现分析
│   ├── lda_trainer.py              # LDA建模（训练/寻优/验证）
│   ├── topic_visualization.py      # 主题可视化页面渲染
│   ├── visualizer.py               # 可视化引擎（词云/热图/PyLDAvis/t-SNE）
│   ├── clustering_module.py        # 聚类分析（K-means/层次/KNN）
│   ├── temporal_analyzer.py        # 时序分析
│   ├── temporal_topic_evolution.py # 主题演变分析
│   ├── comparative_analyzer.py     # 比较分析
│   ├── citation_analyzer.py       # 引用分析
│   ├── semantic_network_builder.py # 语义网络构建（社区/中心性）
│   ├── semantic_network.py         # 语义网络可视化
│   ├── qualitative_coding.py       # 质性编码
│   ├── comprehensive_report.py      # 综合报告生成
│   ├── result_exporter.py          # 结果导出（图表/数据/打包）
│   ├── exporter.py                 # 导出辅助工具
│   ├── database.py                 # 数据库操作封装
│   └── logger.py                   # 日志管理
│
├── utils/
│   ├── session_state.py            # 会话状态管理
│   ├── font_config.py              # 中文字体配置
│   ├── data_export.py              # 数据导出工具
│   └── db_manager.py               # 数据库连接管理
│
├── fonts/                          # 中文字体资源
├── models/                         # 训练模型存储（运行时创建）
└── temp/                           # 临时文件目录
```

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| Web框架 | Streamlit ≥ 1.22.0 |
| 中文分词 | jieba ≥ 0.42.1 + 自定义词典干预 |
| 主题模型 | Gensim ≥ 4.2.0 (Online Variational Bayes) |
| 机器学习 | scikit-learn (K-means/t-SNE/KNN/TF-IDF) |
| 网络分析 | NetworkX ≥ 2.8.0 (社区检测/中心性) |
| 可视化 | Plotly + Matplotlib + PyLDAvis + WordCloud |
| 数据处理 | Pandas + NumPy |
| 数据库 | PyMySQL + MySQL 8.0 |

## 📝 注意事项

- 推荐使用5-50个政策文本进行分析，主题数量一般设置为4-10个
- 页面刷新会导致数据丢失，请及时导出结果
- 大规模文本分析(>500文档)可能需要较长处理时间
- 显示网络图需要安装 **plotly** 和 **networkx** 库
- 词云中文显示需要中文字体文件放入 `fonts/` 目录

## 📄 许可证

MIT License
