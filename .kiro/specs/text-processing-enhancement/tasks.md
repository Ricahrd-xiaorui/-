# Implementation Plan: 学术研究文本分析功能增强

## Overview

本实现计划将学术研究文本分析模块逐步集成到现有的政策文件LDA主题模型分析系统中。

**UI结构（已更新）：**
```
📊 政策文件LDA主题模型可视化分析系统 (v2.0.0)
├── 📁 数据加载
├── ⚙️ 文本预处理 ← 整合专业词典管理
├── 📈 基础文本分析 ← 新增（文本统计、词频、共现）
├── 🎯 主题建模
├── 📊 主题可视化
├── 🔬 高级研究分析 ← 聚类、时序、比较、引用、语义、编码
└── 💾 结果导出
```

## Tasks

- [x] 1. 项目基础设施准备
  - [x] 1.1 扩展会话状态管理
    - 在 utils/session_state.py 中添加新模块所需的会话状态变量
    - _Requirements: All_
  - [x] 1.2 更新主应用入口和UI结构
    - 重构 app.py，实现新的7标签页结构
    - 添加「基础文本分析」标签页
    - 精简「高级研究分析」标签页
    - _Requirements: All_

- [x] 2. 专业词典管理模块（已整合到文本预处理）
  - [x] 2.1 实现词典核心类
    - 创建 modules/dictionary_manager.py
    - 实现 Dictionary 类、DictionaryManager 类
    - _Requirements: 9.1, 9.3, 9.4, 9.5, 9.6_
  - [x] 2.2 编写词典导入导出属性测试
    - **Property 15: 词典导入导出一致性**
    - **Validates: Requirements 9.1, 9.9, 9.10**
  - [x] 2.3 编写词典分词效果属性测试
    - **Property 16: 词典分词效果**
    - **Validates: Requirements 9.3**
  - [x] 2.4 实现词典管理UI
    - 实现 render_dictionary_manager() 完整版
    - 实现 render_dictionary_manager_compact() 紧凑版（用于文本预处理页面）
    - _Requirements: 9.2, 9.7, 9.8, 9.9, 9.10_
  - [x] 2.5 编写术语频率统计属性测试
    - **Property 17: 术语频率统计**
    - **Validates: Requirements 9.8**
  - [x] 2.6 整合到文本预处理模块
    - 在 text_processor.py 中添加词典管理折叠面板
    - 词典管理放在预处理参数设置之前
    - _Requirements: 9.3_

- [x] 3. 文本统计与可读性分析模块（位于基础文本分析）
  - [x] 3.1 实现文本统计核心类
    - 创建 modules/text_statistics.py
    - 实现 TextStatistics 类、MultiDocStatistics 类
    - _Requirements: 8.1, 8.2, 8.3, 8.5, 8.6_
  - [x] 3.2 编写文本统计属性测试
    - **Property 13: 文本统计一致性**
    - **Property 14: TTR范围**
    - **Validates: Requirements 8.1, 8.2**
  - [x] 3.3 实现文本统计UI
    - 实现 render_text_statistics() 函数
    - 包括仪表盘展示、雷达图对比、CSV导出
    - _Requirements: 8.4, 8.7_

- [x] 4. Checkpoint - 基础模块验证
  - 确保词典管理和文本统计模块正常工作
  - 运行所有属性测试，确保通过
  - UI重组完成验证

- [x] 5. 词频与共现分析模块（位于基础文本分析）
  - [x] 5.1 实现词频分析核心类
    - 创建 modules/frequency_analyzer.py
    - 实现 FrequencyAnalyzer 类（词频统计、词性筛选）
    - _Requirements: 2.1, 2.2_
  - [x] 5.2 编写词频统计属性测试

    - **Property 2: 词频统计正确性**
    - **Validates: Requirements 2.1, 2.2**
  - [x] 5.3 实现共现分析核心类
    - 实现 CooccurrenceAnalyzer 类（共现计算、阈值过滤）
    - 实现网络数据转换方法
    - _Requirements: 2.3, 2.5_
  - [x] 5.4 编写共现阈值过滤属性测试

    - **Property 3: 共现频率阈值过滤**
    - **Validates: Requirements 2.5**
  - [x] 5.5 实现词频共现UI
    - 实现 render_frequency_analyzer() 函数
    - 实现 render_cooccurrence_analyzer() 函数（可选，或整合到词频分析中）
    - 包括词频表格、共现网络图、交互式操作、CSV导出
    - _Requirements: 2.4, 2.6, 2.7_

- [x] 6. Checkpoint - 基础文本分析完成验证
  - 确保「基础文本分析」标签页的三个子功能正常工作
  - 运行所有属性测试，确保通过
  - 如有问题请询问用户

- [x] 7. 文本聚类与分类模块（位于高级研究分析）
  - [x] 7.1 实现聚类核心类
    - 创建 modules/clustering_module.py
    - 实现 TextClusterer 类（K-means、层次聚类）
    - 整合原可视化模块中的文档聚类功能
    - _Requirements: 3.1, 3.2, 3.4_
  - [x] 7.2 编写聚类结果属性测试

    - **Property 4: 聚类结果完整性**
    - **Validates: Requirements 3.2, 3.4**
  - [x] 7.3 实现分类核心类
    - 实现 TextClassifier 类（标签管理、自动分类）
    - _Requirements: 3.5, 3.6_
  - [x] 7.4 编写分类标签属性测试

    - **Property 5: 分类标签覆盖**
    - **Validates: Requirements 3.6**
  - [x] 7.5 实现聚类分类UI
    - 实现 render_clustering_module() 函数
    - 包括散点图、树状图、分类标注界面、结果导出
    - _Requirements: 3.3, 3.7_
+
- [x] 8. 时序演变分析模块（位于高级研究分析）
  - [x] 8.1 实现时序分析核心类
    - 创建 modules/temporal_analyzer.py
    - 实现 TemporalAnalyzer 类（时间标签、关键词趋势、主题演变）
    - _Requirements: 4.1, 4.2, 4.3, 4.5_
  - [x] 8.2 编写时序排序属性测试

    - **Property 6: 时序排序正确性**
    - **Validates: Requirements 4.2**
  - [x] 8.3 实现时序分析UI
    - 实现 render_temporal_analyzer() 函数
    - 包括时间标签设置、折线图、堆叠面积图、CSV导出
    - _Requirements: 4.4, 4.6, 4.7_

- [x] 9. 文本比较分析模块（位于高级研究分析）
  - [x] 9.1 实现比较分析核心类
    - 创建 modules/comparative_analyzer.py
    - 实现 ComparativeAnalyzer 类（相似度计算、关键词对比、相似段落）
    - _Requirements: 5.2, 5.3, 5.6_
  - [x] 9.2 编写相似度计算属性测试

    - **Property 7: 相似度计算对称性**
    - **Validates: Requirements 5.2**
  - [ ]* 9.3 编写共同关键词属性测试
    - **Property 8: 共同关键词验证**
    - **Validates: Requirements 5.3**
  - [x] 9.4 实现比较分析UI
    - 实现 render_comparative_analyzer() 函数
    - 包括文档选择、相似度热图、韦恩图、并排对比、结果导出
    - _Requirements: 5.1, 5.4, 5.5, 5.7_

- [x] 10. Checkpoint - 高级分析模块验证（第一批）
  - 确保聚类分类、时序分析、比较分析模块正常工作
  - 运行所有属性测试，确保通过
  - 如有问题请询问用户

- [x] 11. 引用与参考分析模块（位于高级研究分析）
  - [x] 11.1 实现引用分析核心类
    - 创建 modules/citation_analyzer.py
    - 实现 CitationAnalyzer 类（引用提取、网络构建、核心文档识别）
    - _Requirements: 6.1, 6.2, 6.3, 6.5, 6.6_
  - [x] 11.2 编写引用网络属性测试

    - **Property 9: 引用网络一致性**
    - **Validates: Requirements 6.3, 6.5**
  - [x] 11.3 编写核心文档属性测试

    - **Property 10: 核心文档排序**
    - **Validates: Requirements 6.6**
  - [x] 11.4 实现引用分析UI
    - 实现 render_citation_analyzer() 函数
    - 包括引用列表、有向图可视化、核心文档展示、数据导出
    - _Requirements: 6.4, 6.7_

- [x] 12. 语义网络分析模块（位于高级研究分析）
  - [x] 12.1 实现语义网络核心类
    - 创建 modules/semantic_network.py
    - 实现 SemanticNetworkBuilder 类（网络构建、社区检测、中心性计算）
    - _Requirements: 7.1, 7.2, 7.3, 7.5, 7.7_
  - [x] 12.2 编写社区覆盖属性测试

    - **Property 11: 语义网络社区覆盖**
    - **Validates: Requirements 7.5**
  - [x] 12.3 编写中心性指标属性测试

    - **Property 12: 中心性指标范围**
    - **Validates: Requirements 7.7**
  - [x] 12.4 实现语义网络UI
    - 实现 render_semantic_network() 函数
    - 包括核心概念设置、力导向图、社区颜色标注、网络数据导出
    - _Requirements: 7.4, 7.6, 7.8_

- [x] 13. 质性研究文本编码模块（位于高级研究分析）
  - [x] 13.1 实现编码核心类
    - 创建 modules/qualitative_coding.py
    - 实现 Code、CodingScheme、CodedSegment 类
    - 实现 QualitativeCoder 类（编码管理、片段标注、统计）
    - _Requirements: 1.1, 1.2, 1.4, 1.5_
  - [x] 13.2 编写编码持久化属性测试

    - **Property 1: 编码数据持久化一致性**
    - **Validates: Requirements 1.6, 1.7**
  - [x] 13.3 实现质性编码UI
    - 实现 render_qualitative_coding() 函数
    - 包括编码体系管理、文本标注界面、高亮显示、统计展示、导出功能
    - _Requirements: 1.3, 1.6, 1.7_

- [x] 14. Checkpoint - 高级分析模块验证（第二批）
  - 确保引用分析、语义网络、质性编码模块正常工作
  - 运行所有属性测试，确保通过
  - 如有问题请询问用户

- [x] 15. Final Checkpoint - 完整系统验证
  - 运行所有属性测试，确保全部通过
  - 测试各模块间的数据流转
  - 测试与现有LDA分析功能的集成
  - 更新README文档
  - 如有问题请询问用户

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties

**模块位置说明：**
- 📚 词典管理 → 文本预处理（影响分词结果）
- 📊 文本统计、🔢 词频分析、🔗 词语共现 → 基础文本分析
- 🎯 聚类分类、📅 时序分析、🔍 比较分析、📖 引用分析、🕸️ 语义网络、🏷️ 质性编码 → 高级研究分析
