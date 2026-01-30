# Design Document: 学术研究文本分析功能增强

## Overview

本设计文档描述了政策文件LDA主题模型分析系统的学术研究文本分析功能增强方案。系统将在现有的数据加载、文本预处理、LDA建模和可视化基础上，新增9个面向学术研究的高级文本分析模块。

设计遵循以下原则：
- **模块化架构**：每个功能作为独立模块，便于维护和扩展
- **与现有系统集成**：复用现有的数据加载、分词和会话状态管理
- **一致的用户体验**：遵循现有UI风格，使用Streamlit组件
- **数据可导出**：所有分析结果支持CSV/JSON格式导出

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         app.py (主入口)                          │
├─────────────────────────────────────────────────────────────────┤
│  现有模块                    │  新增模块                         │
│  ├── data_loader.py         │  ├── qualitative_coding.py       │
│  ├── text_processor.py      │  ├── frequency_analyzer.py       │
│  ├── model_trainer.py       │  ├── clustering_module.py        │
│  ├── visualizer.py          │  ├── temporal_analyzer.py        │
│  └── exporter.py            │  ├── comparative_analyzer.py     │
│                              │  ├── citation_analyzer.py        │
│                              │  ├── semantic_network.py         │
│                              │  ├── text_statistics.py          │
│                              │  └── dictionary_manager.py       │
├─────────────────────────────────────────────────────────────────┤
│                    utils/session_state.py                        │
│                    (会话状态管理 - 扩展)                          │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. 质性编码模块 (qualitative_coding.py)

```python
class CodingScheme:
    """编码方案类"""
    def __init__(self):
        self.codes: Dict[str, Code] = {}  # 编码字典
        self.hierarchy: Dict[str, List[str]] = {}  # 层级关系
    
    def add_code(self, name: str, description: str, color: str, parent: str = None) -> Code
    def remove_code(self, name: str) -> bool
    def get_children(self, name: str) -> List[Code]
    def to_dict(self) -> dict
    def from_dict(self, data: dict) -> None

class Code:
    """编码类"""
    name: str
    description: str
    color: str
    parent: Optional[str]

class CodedSegment:
    """已编码文本片段"""
    document_id: str
    start_pos: int
    end_pos: int
    text: str
    codes: List[str]

class QualitativeCoder:
    """质性编码器"""
    def __init__(self, scheme: CodingScheme):
        self.scheme = scheme
        self.segments: List[CodedSegment] = []
    
    def add_segment(self, doc_id: str, start: int, end: int, text: str, codes: List[str]) -> CodedSegment
    def get_segments_by_code(self, code_name: str) -> List[CodedSegment]
    def get_code_statistics(self) -> Dict[str, CodeStats]
    def export_to_csv(self) -> str
    def save_scheme(self, filepath: str) -> bool
    def load_scheme(self, filepath: str) -> bool

def render_qualitative_coding():
    """渲染质性编码模块UI"""
```

### 2. 词频与共现分析模块 (frequency_analyzer.py)

```python
class FrequencyAnalyzer:
    """词频分析器"""
    def __init__(self, texts: List[List[str]], pos_tags: List[List[str]] = None):
        self.texts = texts
        self.pos_tags = pos_tags
    
    def calculate_word_frequency(self) -> Dict[str, int]
    def filter_by_pos(self, pos_list: List[str]) -> Dict[str, int]
    def get_top_words(self, n: int) -> List[Tuple[str, int]]

class CooccurrenceAnalyzer:
    """共现分析器"""
    def __init__(self, texts: List[List[str]], window_size: int = 5):
        self.texts = texts
        self.window_size = window_size
        self.cooccurrence_matrix: Dict[Tuple[str, str], int] = {}
    
    def calculate_cooccurrence(self) -> Dict[Tuple[str, str], int]
    def filter_by_threshold(self, min_freq: int) -> Dict[Tuple[str, str], int]
    def to_network_data(self) -> Tuple[List[dict], List[dict]]  # nodes, edges
    def export_matrix_csv(self) -> str

def render_frequency_analyzer():
    """渲染词频与共现分析模块UI"""
```

### 3. 文本聚类与分类模块 (clustering_module.py)

```python
class TextClusterer:
    """文本聚类器"""
    def __init__(self, doc_vectors: np.ndarray, file_names: List[str]):
        self.doc_vectors = doc_vectors
        self.file_names = file_names
    
    def kmeans_clustering(self, n_clusters: int) -> np.ndarray
    def hierarchical_clustering(self, n_clusters: int) -> np.ndarray
    def get_cluster_keywords(self, cluster_id: int, texts: List[List[str]], top_n: int = 10) -> List[str]
    def export_results(self) -> str

class TextClassifier:
    """文本分类器"""
    def __init__(self, doc_vectors: np.ndarray, file_names: List[str]):
        self.doc_vectors = doc_vectors
        self.file_names = file_names
        self.labels: Dict[str, str] = {}  # doc_name -> label
    
    def add_label(self, doc_name: str, label: str) -> None
    def train_classifier(self) -> None
    def predict_unlabeled(self) -> Dict[str, str]

def render_clustering_module():
    """渲染聚类与分类模块UI"""
```

### 4. 时序演变分析模块 (temporal_analyzer.py)

```python
class TemporalAnalyzer:
    """时序分析器"""
    def __init__(self, texts: List[List[str]], file_names: List[str]):
        self.texts = texts
        self.file_names = file_names
        self.time_labels: Dict[str, str] = {}  # doc_name -> time_label
    
    def set_time_label(self, doc_name: str, time_label: str) -> None
    def get_documents_by_period(self, period: str) -> List[str]
    def analyze_keyword_trend(self, keyword: str) -> Dict[str, int]
    def analyze_topic_evolution(self, doc_topic_dist: np.ndarray) -> Dict[str, List[float]]
    def export_trend_data(self) -> str

def render_temporal_analyzer():
    """渲染时序分析模块UI"""
```

### 5. 文本比较分析模块 (comparative_analyzer.py)

```python
class ComparativeAnalyzer:
    """比较分析器"""
    def __init__(self, texts: List[List[str]], file_names: List[str]):
        self.texts = texts
        self.file_names = file_names
    
    def calculate_similarity(self, doc1_idx: int, doc2_idx: int, method: str = 'cosine') -> float
    def calculate_similarity_matrix(self, method: str = 'cosine') -> np.ndarray
    def find_common_keywords(self, doc_indices: List[int], top_n: int = 20) -> List[str]
    def find_unique_keywords(self, doc_idx: int, other_indices: List[int], top_n: int = 20) -> List[str]
    def find_similar_segments(self, doc1_idx: int, doc2_idx: int, threshold: float = 0.8) -> List[Tuple[str, str, float]]
    def export_comparison(self) -> str

def render_comparative_analyzer():
    """渲染比较分析模块UI"""
```

### 6. 引用与参考分析模块 (citation_analyzer.py)

```python
class CitationAnalyzer:
    """引用分析器"""
    def __init__(self, raw_texts: List[str], file_names: List[str]):
        self.raw_texts = raw_texts
        self.file_names = file_names
        self.citations: Dict[str, List[str]] = {}  # doc_name -> cited_docs
    
    def extract_citations(self) -> Dict[str, List[str]]
    def build_citation_network(self) -> nx.DiGraph
    def get_citation_count(self, doc_name: str) -> Tuple[int, int]  # (cited_by, cites)
    def find_core_documents(self, top_n: int = 5) -> List[Tuple[str, int]]
    def export_network_data(self) -> str

def render_citation_analyzer():
    """渲染引用分析模块UI"""
```

### 7. 语义网络分析模块 (semantic_network.py)

```python
class SemanticNetworkBuilder:
    """语义网络构建器"""
    def __init__(self, texts: List[List[str]], cooccurrence_data: Dict[Tuple[str, str], int]):
        self.texts = texts
        self.cooccurrence_data = cooccurrence_data
        self.network: nx.Graph = None
    
    def build_network(self, min_weight: int = 2) -> nx.Graph
    def filter_by_center(self, center_word: str, max_depth: int = 2) -> nx.Graph
    def detect_communities(self) -> Dict[str, int]  # node -> community_id
    def calculate_centrality(self) -> Dict[str, Dict[str, float]]  # node -> {metric: value}
    def export_network(self) -> Tuple[str, str]  # nodes_csv, edges_csv

def render_semantic_network():
    """渲染语义网络模块UI"""
```

### 8. 文本统计与可读性分析模块 (text_statistics.py)

```python
class TextStatistics:
    """文本统计分析器"""
    def __init__(self, raw_text: str, tokenized_text: List[str]):
        self.raw_text = raw_text
        self.tokenized_text = tokenized_text
    
    def count_characters(self) -> int
    def count_words(self) -> int
    def count_sentences(self) -> int
    def count_paragraphs(self) -> int
    def calculate_avg_sentence_length(self) -> float
    def calculate_avg_word_length(self) -> float
    def calculate_ttr(self) -> float  # Type-Token Ratio
    def calculate_readability_index(self) -> float
    def to_dict(self) -> dict

class MultiDocStatistics:
    """多文档统计对比"""
    def __init__(self, documents: List[TextStatistics], file_names: List[str]):
        self.documents = documents
        self.file_names = file_names
    
    def compare_statistics(self) -> pd.DataFrame
    def export_comparison(self) -> str

def render_text_statistics():
    """渲染文本统计模块UI"""
```

### 9. 专业词典管理模块 (dictionary_manager.py)

```python
class DictionaryManager:
    """专业词典管理器"""
    def __init__(self):
        self.dictionaries: Dict[str, Dictionary] = {}
        self.active_dictionaries: List[str] = []
    
    def import_dictionary(self, filepath: str, name: str) -> bool
    def create_dictionary(self, name: str) -> Dictionary
    def add_word(self, dict_name: str, word: str, pos: str = None) -> bool
    def remove_word(self, dict_name: str, word: str) -> bool
    def activate_dictionary(self, name: str) -> None
    def deactivate_dictionary(self, name: str) -> None
    def apply_to_jieba(self) -> None
    def find_terms_in_text(self, text: str) -> List[Tuple[str, int, int]]  # (term, start, end)
    def count_term_frequency(self, texts: List[str]) -> Dict[str, int]
    def export_dictionary(self, name: str, filepath: str) -> bool
    def save_all(self, filepath: str) -> bool
    def load_all(self, filepath: str) -> bool

class Dictionary:
    """词典类"""
    name: str
    words: Dict[str, str]  # word -> pos_tag
    
    def add(self, word: str, pos: str = None) -> None
    def remove(self, word: str) -> bool
    def contains(self, word: str) -> bool
    def to_list(self) -> List[Tuple[str, str]]

def render_dictionary_manager():
    """渲染词典管理模块UI"""
```

## Data Models

### 会话状态扩展

```python
# 在 utils/session_state.py 中扩展
EXTENDED_SESSION_STATE = {
    # 质性编码
    'coding_scheme': None,  # CodingScheme实例
    'coded_segments': [],   # List[CodedSegment]
    
    # 词频共现
    'word_frequency': {},   # Dict[str, int]
    'cooccurrence_matrix': {},  # Dict[Tuple[str, str], int]
    
    # 聚类分类
    'cluster_labels': None,  # np.ndarray
    'classification_labels': {},  # Dict[str, str]
    
    # 时序分析
    'time_labels': {},  # Dict[str, str]
    
    # 引用分析
    'citation_network': None,  # nx.DiGraph
    
    # 语义网络
    'semantic_network': None,  # nx.Graph
    'community_labels': {},  # Dict[str, int]
    
    # 专业词典
    'dictionary_manager': None,  # DictionaryManager实例
}
```

### 数据导出格式

**编码结果CSV格式：**
```csv
编码,文本片段,来源文档,起始位置,结束位置
创新政策,加快推进科技创新,政策1.txt,100,108
```

**共现矩阵CSV格式：**
```csv
词语1,词语2,共现频率
创新,发展,25
政策,实施,18
```

**聚类结果CSV格式：**
```csv
文档名,聚类ID,主导主题
政策1.txt,1,主题3
政策2.txt,2,主题1
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: 编码数据持久化一致性
*For any* 编码方案和已编码片段集合，保存到文件后再加载，应该得到与原始数据等价的编码方案和片段集合。
**Validates: Requirements 1.6, 1.7**

### Property 2: 词频统计正确性
*For any* 分词后的文本集合，词频统计结果中所有词频之和应等于文本中的总词数，且按词性筛选后的结果中每个词都属于指定词性。
**Validates: Requirements 2.1, 2.2**

### Property 3: 共现频率阈值过滤
*For any* 共现分析结果和最小频率阈值，过滤后的结果中每对词语的共现频率都应大于等于阈值。
**Validates: Requirements 2.5**

### Property 4: 聚类结果完整性
*For any* 文档集合和指定的聚类数量K，聚类结果应将所有文档分配到K个聚类中，且每个聚类都有代表性关键词。
**Validates: Requirements 3.2, 3.4**

### Property 5: 分类标签覆盖
*For any* 部分标注的文档集合，自动分类后所有文档都应有分类标签。
**Validates: Requirements 3.6**

### Property 6: 时序排序正确性
*For any* 带有时间标签的文档集合，按时间排序后文档应按时间标签的升序排列。
**Validates: Requirements 4.2**

### Property 7: 相似度计算对称性
*For any* 两个文档A和B，similarity(A, B) 应等于 similarity(B, A)，且 similarity(A, A) 应等于 1.0。
**Validates: Requirements 5.2**

### Property 8: 共同关键词验证
*For any* 被识别为共同关键词的词语，该词语应在所有参与比较的文档中都出现。
**Validates: Requirements 5.3**

### Property 9: 引用网络一致性
*For any* 引用网络，文档的被引用次数应等于网络中指向该文档的入边数量，引用次数应等于出边数量。
**Validates: Requirements 6.3, 6.5**

### Property 10: 核心文档排序
*For any* 引用网络中识别的核心文档列表，列表应按被引用次数降序排列。
**Validates: Requirements 6.6**

### Property 11: 语义网络社区覆盖
*For any* 语义网络的社区检测结果，网络中的每个节点都应被分配到一个社区。
**Validates: Requirements 7.5**

### Property 12: 中心性指标范围
*For any* 语义网络的度中心性计算结果，所有节点的度中心性值应在[0, 1]范围内。
**Validates: Requirements 7.7**

### Property 13: 文本统计一致性
*For any* 文本，字符数应大于等于词语数，词语数应大于等于句子数，句子数应大于等于段落数。
**Validates: Requirements 8.1**

### Property 14: TTR范围
*For any* 文本的词汇丰富度(TTR)计算结果，TTR值应在(0, 1]范围内。
**Validates: Requirements 8.2**

### Property 15: 词典导入导出一致性
*For any* 专业词典，导出到文件后再导入，应得到包含相同词汇的词典。
**Validates: Requirements 9.1, 9.9, 9.10**

### Property 16: 词典分词效果
*For any* 导入的专业词典词汇，在包含该词汇的文本上进行分词，分词结果应包含该词汇作为独立词语。
**Validates: Requirements 9.3**

### Property 17: 术语频率统计
*For any* 专业词典和文本集合，术语频率统计结果中每个术语的频率应等于该术语在所有文本中出现的实际次数。
**Validates: Requirements 9.8**

## Error Handling

### 输入验证
- 空文本检查：所有分析模块在处理前检查输入文本是否为空
- 参数范围检查：聚类数量、阈值等参数需在有效范围内
- 文件格式检查：导入词典和编码方案时验证文件格式

### 异常处理
- 分词失败：捕获jieba分词异常，返回空列表并记录日志
- 网络构建失败：节点或边数据异常时返回空网络
- 导出失败：文件写入失败时显示错误提示

### 用户反馈
- 使用Streamlit的st.warning()和st.error()显示错误信息
- 使用log_message()记录详细错误日志
- 长时间操作显示进度条和状态文本

## Testing Strategy

### 单元测试
- 使用pytest框架
- 测试各模块的核心计算逻辑
- 测试边界条件和异常情况

### 属性测试
- 使用hypothesis库进行属性测试
- 每个属性测试运行至少100次迭代
- 测试标注格式：**Feature: text-processing-enhancement, Property N: {property_text}**

### 集成测试
- 测试模块间的数据流转
- 测试与现有系统的集成
- 测试UI交互流程
