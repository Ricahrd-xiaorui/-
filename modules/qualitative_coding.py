# -*- coding: utf-8 -*-
"""
质性研究文本编码模块 (Qualitative Research Text Coding Module)

本模块提供质性研究的文本编码功能，包括：
- 编码体系的创建和管理（支持层级结构）
- 文本片段的编码标注
- 编码统计分析
- 编码结果的导入导出

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7
"""

import os
import json
import csv
from io import StringIO
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
from datetime import datetime


@dataclass
class Code:
    """
    编码类 - 表示单个编码
    
    Attributes:
        name: 编码名称（唯一标识）
        description: 编码描述
        color: 编码颜色（用于高亮显示，如 #FF5733）
        parent: 父编码名称（用于层级结构，None表示顶级编码）
    
    Requirements: 1.1
    """
    name: str
    description: str = ""
    color: str = "#3498db"  # 默认蓝色
    parent: Optional[str] = None
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'name': self.name,
            'description': self.description,
            'color': self.color,
            'parent': self.parent
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Code':
        """从字典创建实例"""
        return cls(
            name=data.get('name', ''),
            description=data.get('description', ''),
            color=data.get('color', '#3498db'),
            parent=data.get('parent')
        )


@dataclass
class CodedSegment:
    """
    已编码文本片段 - 表示被标注的文本片段
    
    Attributes:
        document_id: 来源文档标识
        start_pos: 片段在文档中的起始位置
        end_pos: 片段在文档中的结束位置
        text: 被编码的文本内容
        codes: 分配给该片段的编码名称列表
        created_at: 创建时间
        note: 备注信息
    
    Requirements: 1.2
    """
    document_id: str
    start_pos: int
    end_pos: int
    text: str
    codes: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    note: str = ""
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'document_id': self.document_id,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'text': self.text,
            'codes': self.codes,
            'created_at': self.created_at,
            'note': self.note
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CodedSegment':
        """从字典创建实例"""
        return cls(
            document_id=data.get('document_id', ''),
            start_pos=data.get('start_pos', 0),
            end_pos=data.get('end_pos', 0),
            text=data.get('text', ''),
            codes=data.get('codes', []),
            created_at=data.get('created_at', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            note=data.get('note', '')
        )
    
    def add_code(self, code_name: str) -> bool:
        """
        为片段添加编码
        
        Args:
            code_name: 编码名称
            
        Returns:
            bool: 是否成功添加
        """
        if code_name and code_name not in self.codes:
            self.codes.append(code_name)
            return True
        return False
    
    def remove_code(self, code_name: str) -> bool:
        """
        从片段移除编码
        
        Args:
            code_name: 编码名称
            
        Returns:
            bool: 是否成功移除
        """
        if code_name in self.codes:
            self.codes.remove(code_name)
            return True
        return False


@dataclass
class CodeStats:
    """
    编码统计信息
    
    Attributes:
        code_name: 编码名称
        frequency: 使用频次
        text_coverage: 覆盖的文本字符数
        document_count: 涉及的文档数
        segments: 相关的片段列表
    """
    code_name: str
    frequency: int = 0
    text_coverage: int = 0
    document_count: int = 0
    segments: List[CodedSegment] = field(default_factory=list)


class CodingScheme:
    """
    编码方案类 - 管理编码体系
    
    支持创建层级结构的编码体系，包括父编码和子编码。
    
    Attributes:
        codes: 编码字典，键为编码名称
        hierarchy: 层级关系，键为父编码名称，值为子编码名称列表
        name: 编码方案名称
        description: 编码方案描述
    
    Requirements: 1.1, 1.4
    """
    
    def __init__(self, name: str = "默认编码方案", description: str = ""):
        """
        初始化编码方案
        
        Args:
            name: 方案名称
            description: 方案描述
        """
        self.codes: Dict[str, Code] = {}
        self.hierarchy: Dict[str, List[str]] = {}  # parent -> [children]
        self.name = name
        self.description = description
    
    def add_code(self, name: str, description: str = "", 
                 color: str = "#3498db", parent: Optional[str] = None) -> Optional[Code]:
        """
        添加编码到方案
        
        Args:
            name: 编码名称
            description: 编码描述
            color: 编码颜色
            parent: 父编码名称（可选）
            
        Returns:
            Optional[Code]: 创建的编码，如果名称已存在则返回None
        
        Requirements: 1.1, 1.4
        """
        name = name.strip()
        if not name or name in self.codes:
            return None
        
        # 验证父编码存在
        if parent and parent not in self.codes:
            return None
        
        code = Code(name=name, description=description, color=color, parent=parent)
        self.codes[name] = code
        
        # 更新层级关系
        if parent:
            if parent not in self.hierarchy:
                self.hierarchy[parent] = []
            self.hierarchy[parent].append(name)
        
        return code
    
    def remove_code(self, name: str) -> bool:
        """
        从方案中移除编码
        
        同时移除该编码的所有子编码。
        
        Args:
            name: 编码名称
            
        Returns:
            bool: 是否成功移除
        """
        if name not in self.codes:
            return False
        
        # 递归移除子编码
        children = self.get_children(name)
        for child in children:
            self.remove_code(child)
        
        # 从父编码的子列表中移除
        code = self.codes[name]
        if code.parent and code.parent in self.hierarchy:
            if name in self.hierarchy[code.parent]:
                self.hierarchy[code.parent].remove(name)
        
        # 移除自身的层级记录
        if name in self.hierarchy:
            del self.hierarchy[name]
        
        # 移除编码
        del self.codes[name]
        return True
    
    def update_code(self, name: str, description: str = None, 
                    color: str = None) -> bool:
        """
        更新编码信息
        
        Args:
            name: 编码名称
            description: 新描述（None表示不更新）
            color: 新颜色（None表示不更新）
            
        Returns:
            bool: 是否成功更新
        """
        if name not in self.codes:
            return False
        
        code = self.codes[name]
        if description is not None:
            code.description = description
        if color is not None:
            code.color = color
        
        return True
    
    def get_code(self, name: str) -> Optional[Code]:
        """
        获取指定编码
        
        Args:
            name: 编码名称
            
        Returns:
            Optional[Code]: 编码实例，不存在则返回None
        """
        return self.codes.get(name)
    
    def get_children(self, name: str) -> List[str]:
        """
        获取编码的所有子编码名称
        
        Args:
            name: 父编码名称
            
        Returns:
            List[str]: 子编码名称列表
        
        Requirements: 1.4
        """
        return self.hierarchy.get(name, []).copy()
    
    def get_all_descendants(self, name: str) -> List[str]:
        """
        获取编码的所有后代编码（递归）
        
        Args:
            name: 编码名称
            
        Returns:
            List[str]: 所有后代编码名称列表
        """
        descendants = []
        children = self.get_children(name)
        for child in children:
            descendants.append(child)
            descendants.extend(self.get_all_descendants(child))
        return descendants
    
    def get_root_codes(self) -> List[str]:
        """
        获取所有顶级编码（无父编码）
        
        Returns:
            List[str]: 顶级编码名称列表
        """
        return [name for name, code in self.codes.items() if code.parent is None]
    
    def get_all_codes(self) -> List[Code]:
        """
        获取所有编码
        
        Returns:
            List[Code]: 编码列表
        """
        return list(self.codes.values())
    
    def get_code_count(self) -> int:
        """
        获取编码总数
        
        Returns:
            int: 编码数量
        """
        return len(self.codes)
    
    def to_dict(self) -> dict:
        """
        将编码方案转换为字典格式（用于序列化）
        
        Returns:
            dict: 编码方案数据
        """
        return {
            'name': self.name,
            'description': self.description,
            'codes': {name: code.to_dict() for name, code in self.codes.items()},
            'hierarchy': self.hierarchy
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CodingScheme':
        """
        从字典数据创建编码方案实例
        
        Args:
            data: 编码方案数据
            
        Returns:
            CodingScheme: 编码方案实例
        """
        scheme = cls(
            name=data.get('name', '默认编码方案'),
            description=data.get('description', '')
        )
        
        # 恢复编码
        for name, code_data in data.get('codes', {}).items():
            scheme.codes[name] = Code.from_dict(code_data)
        
        # 恢复层级关系
        scheme.hierarchy = data.get('hierarchy', {})
        
        return scheme



class QualitativeCoder:
    """
    质性编码器 - 管理文本编码过程
    
    提供完整的质性编码功能，包括：
    - 编码方案管理
    - 文本片段标注
    - 编码统计分析
    - 结果导入导出
    
    Attributes:
        scheme: 编码方案
        segments: 已编码的文本片段列表
    
    Requirements: 1.1-1.7
    """
    
    def __init__(self, scheme: CodingScheme = None):
        """
        初始化质性编码器
        
        Args:
            scheme: 编码方案，如果为None则创建默认方案
        """
        self.scheme = scheme if scheme else CodingScheme()
        self.segments: List[CodedSegment] = []
    
    def add_segment(self, doc_id: str, start: int, end: int, 
                    text: str, codes: List[str], note: str = "") -> Optional[CodedSegment]:
        """
        添加已编码的文本片段
        
        Args:
            doc_id: 文档标识
            start: 起始位置
            end: 结束位置
            text: 文本内容
            codes: 编码名称列表
            note: 备注信息
            
        Returns:
            Optional[CodedSegment]: 创建的片段，如果参数无效则返回None
        
        Requirements: 1.2
        """
        if not doc_id or start < 0 or end <= start or not text:
            return None
        
        # 验证编码存在
        valid_codes = [c for c in codes if c in self.scheme.codes]
        
        segment = CodedSegment(
            document_id=doc_id,
            start_pos=start,
            end_pos=end,
            text=text,
            codes=valid_codes,
            note=note
        )
        
        self.segments.append(segment)
        return segment
    
    def remove_segment(self, index: int) -> bool:
        """
        移除指定索引的片段
        
        Args:
            index: 片段索引
            
        Returns:
            bool: 是否成功移除
        """
        if 0 <= index < len(self.segments):
            self.segments.pop(index)
            return True
        return False
    
    def get_segment(self, index: int) -> Optional[CodedSegment]:
        """
        获取指定索引的片段
        
        Args:
            index: 片段索引
            
        Returns:
            Optional[CodedSegment]: 片段实例
        """
        if 0 <= index < len(self.segments):
            return self.segments[index]
        return None
    
    def get_segments_by_code(self, code_name: str, include_children: bool = False) -> List[CodedSegment]:
        """
        获取使用指定编码的所有片段
        
        Args:
            code_name: 编码名称
            include_children: 是否包含子编码的片段
            
        Returns:
            List[CodedSegment]: 片段列表
        
        Requirements: 1.5
        """
        target_codes = {code_name}
        
        if include_children:
            descendants = self.scheme.get_all_descendants(code_name)
            target_codes.update(descendants)
        
        return [seg for seg in self.segments 
                if any(c in target_codes for c in seg.codes)]
    
    def get_segments_by_document(self, doc_id: str) -> List[CodedSegment]:
        """
        获取指定文档的所有片段
        
        Args:
            doc_id: 文档标识
            
        Returns:
            List[CodedSegment]: 片段列表
        """
        return [seg for seg in self.segments if seg.document_id == doc_id]
    
    def get_code_statistics(self) -> Dict[str, CodeStats]:
        """
        获取所有编码的统计信息
        
        Returns:
            Dict[str, CodeStats]: 编码名称到统计信息的映射
        
        Requirements: 1.5
        """
        stats = {}
        
        for code_name in self.scheme.codes:
            segments = self.get_segments_by_code(code_name)
            
            # 计算统计数据
            frequency = len(segments)
            text_coverage = sum(len(seg.text) for seg in segments)
            documents = set(seg.document_id for seg in segments)
            
            stats[code_name] = CodeStats(
                code_name=code_name,
                frequency=frequency,
                text_coverage=text_coverage,
                document_count=len(documents),
                segments=segments
            )
        
        return stats
    
    def get_document_statistics(self) -> Dict[str, dict]:
        """
        获取按文档分组的统计信息
        
        Returns:
            Dict[str, dict]: 文档ID到统计信息的映射
        """
        doc_stats = defaultdict(lambda: {
            'segment_count': 0,
            'code_count': 0,
            'codes_used': set(),
            'text_coverage': 0
        })
        
        for seg in self.segments:
            doc_id = seg.document_id
            doc_stats[doc_id]['segment_count'] += 1
            doc_stats[doc_id]['code_count'] += len(seg.codes)
            doc_stats[doc_id]['codes_used'].update(seg.codes)
            doc_stats[doc_id]['text_coverage'] += len(seg.text)
        
        # 转换set为list以便序列化
        result = {}
        for doc_id, stats in doc_stats.items():
            result[doc_id] = {
                'segment_count': stats['segment_count'],
                'code_count': stats['code_count'],
                'codes_used': list(stats['codes_used']),
                'text_coverage': stats['text_coverage']
            }
        
        return result
    
    def export_to_csv(self) -> str:
        """
        导出编码结果为CSV格式
        
        Returns:
            str: CSV格式的编码结果
        
        Requirements: 1.6
        """
        output = StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)
        
        # 写入表头
        writer.writerow(['编码', '文本片段', '来源文档', '起始位置', '结束位置', '备注', '创建时间'])
        
        # 写入数据
        for seg in self.segments:
            for code in seg.codes:
                writer.writerow([
                    code,
                    seg.text,
                    seg.document_id,
                    seg.start_pos,
                    seg.end_pos,
                    seg.note,
                    seg.created_at
                ])
        
        return output.getvalue()
    
    def export_statistics_csv(self) -> str:
        """
        导出编码统计为CSV格式
        
        Returns:
            str: CSV格式的统计数据
        """
        output = StringIO()
        writer = csv.writer(output)
        
        # 写入表头
        writer.writerow(['编码名称', '使用频次', '覆盖文本量', '涉及文档数', '父编码'])
        
        # 获取统计数据
        stats = self.get_code_statistics()
        
        # 写入数据
        for code_name, code_stats in stats.items():
            code = self.scheme.get_code(code_name)
            parent = code.parent if code else ""
            writer.writerow([
                code_name,
                code_stats.frequency,
                code_stats.text_coverage,
                code_stats.document_count,
                parent or ""
            ])
        
        return output.getvalue()
    
    def save_scheme(self, filepath: str) -> bool:
        """
        保存编码方案到文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 是否成功保存
        
        Requirements: 1.7
        """
        try:
            data = {
                'scheme': self.scheme.to_dict(),
                'segments': [seg.to_dict() for seg in self.segments],
                'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 确保目录存在
            dir_path = os.path.dirname(filepath)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception:
            return False
    
    def load_scheme(self, filepath: str) -> bool:
        """
        从文件加载编码方案
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 是否成功加载
        
        Requirements: 1.7
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载编码方案
            self.scheme = CodingScheme.from_dict(data.get('scheme', {}))
            
            # 加载片段
            self.segments = [
                CodedSegment.from_dict(seg_data) 
                for seg_data in data.get('segments', [])
            ]
            
            return True
        except Exception:
            return False
    
    def to_dict(self) -> dict:
        """
        将编码器状态转换为字典（用于序列化）
        
        Returns:
            dict: 编码器状态数据
        """
        return {
            'scheme': self.scheme.to_dict(),
            'segments': [seg.to_dict() for seg in self.segments]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'QualitativeCoder':
        """
        从字典数据创建编码器实例
        
        Args:
            data: 编码器状态数据
            
        Returns:
            QualitativeCoder: 编码器实例
        """
        scheme = CodingScheme.from_dict(data.get('scheme', {}))
        coder = cls(scheme=scheme)
        
        coder.segments = [
            CodedSegment.from_dict(seg_data)
            for seg_data in data.get('segments', [])
        ]
        
        return coder
    
    def clear_segments(self) -> None:
        """清除所有已编码片段"""
        self.segments = []
    
    def get_segment_count(self) -> int:
        """获取片段总数"""
        return len(self.segments)
    
    def get_coded_documents(self) -> List[str]:
        """获取所有已编码的文档ID列表"""
        return list(set(seg.document_id for seg in self.segments))



# ============================================================================
# Streamlit UI 渲染函数
# ============================================================================

def render_qualitative_coding():
    """
    渲染质性编码模块UI
    
    Requirements: 1.1-1.7
    """
    import streamlit as st
    import pandas as pd
    from utils.session_state import log_message
    
    st.header("🏷️ 质性研究文本编码")
    
    # 功能介绍与操作手册
    with st.expander("📖 功能介绍与操作手册", expanded=False):
        st.markdown("""
        ## 🏷️ 质性研究文本编码模块
        
        **功能概述**：支持研究者对文本进行主题编码和归类分析，是质性研究中内容分析的核心工具。
        
        ---
        
        ### 🎯 核心功能
        
        | 功能 | 说明 | 应用场景 |
        |------|------|----------|
        | 编码体系管理 | 创建、编辑、删除编码 | 建立分析框架 |
        | 层级编码 | 支持父编码-子编码结构 | 多层次主题分类 |
        | 文本标注 | 选中文本片段分配编码 | 内容分析、主题提取 |
        | 高亮显示 | 以颜色标注已编码片段 | 可视化编码结果 |
        | 统计分析 | 编码频次和覆盖率统计 | 量化质性数据 |
        | 导入导出 | 保存/加载编码方案和结果 | 团队协作、结果复用 |
        
        ---
        
        ### 📋 操作步骤
        
        **1. 创建编码体系**
        - 进入「编码体系管理」标签页
        - 输入编码名称、描述
        - 选择编码颜色（用于高亮显示）
        - 可选择父编码创建层级结构
        - 点击「添加编码」
        
        **2. 文本标注**
        - 进入「文本标注」标签页
        - 选择要标注的文档
        - 在文本中选择片段（输入起始和结束位置）
        - 选择要分配的编码（可多选）
        - 点击「添加标注」
        
        **3. 查看统计**
        - 进入「编码统计」标签页
        - 查看各编码的使用频次
        - 查看编码覆盖的文本量
        - 可视化编码分布
        
        **4. 保存与导出**
        - 进入「保存/加载」标签页
        - 保存编码方案（JSON格式）
        - 导出编码结果（CSV格式）
        - 可加载之前保存的编码方案
        
        ---
        
        ### 💡 使用建议
        
        **编码体系设计**
        - 先阅读部分文本，初步确定编码类别
        - 编码应互斥且穷尽（MECE原则）
        - 使用层级结构组织复杂的编码体系
        - 为每个编码写清晰的描述，便于一致性编码
        
        **编码过程**
        - 建议先进行试编码，检验编码体系
        - 同一片段可分配多个编码
        - 定期检查编码一致性
        - 保存编码方案以便团队共享
        
        **学术研究建议**
        - 记录编码决策过程
        - 计算编码者间信度（如需多人编码）
        - 导出结果用于后续统计分析
        - 在论文中报告编码体系和过程
        
        ---
        
        ### 📁 导出格式
        
        **编码方案（JSON）**
        - 包含所有编码定义
        - 包含层级关系
        - 可在其他项目中复用
        
        **编码结果（CSV）**
        - 编码名称
        - 文本片段
        - 来源文档
        - 起始/结束位置
        
        **统计数据（CSV）**
        - 编码频次
        - 覆盖文本量
        - 百分比分布
        """)
    
    # 初始化编码器
    if st.session_state.get("coding_scheme") is None:
        st.session_state["coding_scheme"] = QualitativeCoder()
    
    coder: QualitativeCoder = st.session_state["coding_scheme"]
    
    # 创建标签页
    tabs = st.tabs(["编码体系管理", "文本标注", "编码统计", "保存/加载"])
    
    # ========== 编码体系管理标签页 ==========
    with tabs[0]:
        _render_coding_scheme_tab(coder)
    
    # ========== 文本标注标签页 ==========
    with tabs[1]:
        _render_text_annotation_tab(coder)
    
    # ========== 编码统计标签页 ==========
    with tabs[2]:
        _render_statistics_tab(coder)
    
    # ========== 保存/加载标签页 ==========
    with tabs[3]:
        _render_save_load_tab(coder)



def _render_coding_scheme_tab(coder: QualitativeCoder):
    """渲染编码体系管理标签页"""
    import streamlit as st
    import pandas as pd
    from utils.session_state import log_message
    
    st.subheader("编码体系管理")
    
    # 方案基本信息
    col1, col2 = st.columns(2)
    with col1:
        new_name = st.text_input("方案名称", value=coder.scheme.name, key="scheme_name")
        if new_name != coder.scheme.name:
            coder.scheme.name = new_name
    with col2:
        new_desc = st.text_input("方案描述", value=coder.scheme.description, key="scheme_desc")
        if new_desc != coder.scheme.description:
            coder.scheme.description = new_desc
    
    st.divider()
    
    # 显示现有编码
    st.markdown("**现有编码**")
    codes = coder.scheme.get_all_codes()
    
    if codes:
        # 构建编码表格数据
        code_data = []
        for code in codes:
            code_data.append({
                "编码名称": code.name,
                "描述": code.description or "-",
                "颜色": code.color,
                "父编码": code.parent or "-",
                "子编码数": len(coder.scheme.get_children(code.name))
            })
        
        df = pd.DataFrame(code_data)
        
        # 使用自定义样式显示颜色
        st.dataframe(df, width='stretch', hide_index=True)
        
        # 编码操作
        st.markdown("**编码操作**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            code_names = [c.name for c in codes]
            selected_code = st.selectbox("选择编码", code_names, key="select_code_manage")
        
        with col2:
            if selected_code:
                code = coder.scheme.get_code(selected_code)
                if code:
                    new_color = st.color_picker("修改颜色", value=code.color, key="edit_color")
                    if new_color != code.color:
                        coder.scheme.update_code(selected_code, color=new_color)
                        st.rerun()
        
        with col3:
            if selected_code:
                st.write("")
                st.write("")
                if st.button("删除编码", key="delete_code", type="secondary"):
                    if coder.scheme.remove_code(selected_code):
                        log_message(f"已删除编码: {selected_code}", level="warning")
                        st.rerun()
    else:
        st.info("暂无编码，请添加编码")
    
    st.divider()
    
    # 添加新编码
    st.markdown("**添加新编码**")
    col1, col2 = st.columns(2)
    with col1:
        new_code_name = st.text_input("编码名称", key="new_code_name")
        new_code_desc = st.text_input("编码描述（可选）", key="new_code_desc")
    with col2:
        new_code_color = st.color_picker("编码颜色", value="#3498db", key="new_code_color")
        
        # 父编码选择
        parent_options = ["无（顶级编码）"] + [c.name for c in codes]
        parent_select = st.selectbox("父编码（可选）", parent_options, key="new_code_parent")
        parent_code = None if parent_select == "无（顶级编码）" else parent_select
    
    if st.button("添加编码", key="add_code"):
        if new_code_name:
            result = coder.scheme.add_code(
                name=new_code_name,
                description=new_code_desc,
                color=new_code_color,
                parent=parent_code
            )
            if result:
                log_message(f"已添加编码: {new_code_name}")
                st.success(f"成功添加编码: {new_code_name}")
                st.rerun()
            else:
                st.error("添加失败，编码名称可能已存在或父编码无效")
        else:
            st.warning("请输入编码名称")



def _render_text_annotation_tab(coder: QualitativeCoder):
    """渲染文本标注标签页"""
    import streamlit as st
    import pandas as pd
    from utils.session_state import log_message
    
    st.subheader("文本标注")
    
    # 检查是否有编码
    codes = coder.scheme.get_all_codes()
    if not codes:
        st.warning("请先在「编码体系管理」中添加编码")
        return
    
    # 检查是否有已加载的文本
    raw_texts = st.session_state.get("raw_texts", [])
    file_names = st.session_state.get("file_names", [])
    
    if not raw_texts:
        st.info("请先在「数据加载」模块中加载文本文件")
        
        # 提供手动输入选项
        st.markdown("**或手动输入文本进行编码：**")
        manual_doc_id = st.text_input("文档标识", value="手动输入", key="manual_doc_id")
        manual_text = st.text_area("输入文本", height=200, key="manual_text_input")
        
        if manual_text:
            _render_annotation_interface(coder, manual_doc_id, manual_text)
        return
    
    # 选择文档
    selected_file = st.selectbox("选择文档", file_names, key="annotation_file_select")
    
    if selected_file:
        file_idx = file_names.index(selected_file)
        doc_text = raw_texts[file_idx]
        
        _render_annotation_interface(coder, selected_file, doc_text)


def _render_annotation_interface(coder: QualitativeCoder, doc_id: str, doc_text: str):
    """渲染标注界面"""
    import streamlit as st
    import pandas as pd
    from utils.session_state import log_message

    codes = coder.scheme.get_all_codes()

    # 获取该文档的已编码片段
    doc_segments = coder.get_segments_by_document(doc_id)

    # 初始化session state
    if 'anno_selected_text' not in st.session_state:
        st.session_state.anno_selected_text = ""
    if 'anno_selected_start' not in st.session_state:
        st.session_state.anno_selected_start = -1
    if 'anno_selected_end' not in st.session_state:
        st.session_state.anno_selected_end = -1

    st.markdown("**文本内容**")

    # 使用text_area显示文本（支持选择），设置较大高度
    st.text_area(
        "文档内容",
        value=doc_text,
        height=300,
        key="doc_text_display",
        label_visibility="collapsed",
        help="在文本区域中拖动鼠标选择文字"
    )

    st.markdown("""
    <style>
    .stTextArea textarea {
        user-select: text !important;
        -webkit-user-select: text !important;
        -moz-user-select: text !important;
        -ms-user-select: text !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("🗑️ 清除选择", key="clear_selection"):
        st.session_state.anno_selected_text = ""
        st.session_state.anno_selected_start = -1
        st.session_state.anno_selected_end = -1

    st.divider()

    # 添加新标注
    st.markdown("**添加标注**")

    # 粘贴选中的文本
    pasted_text = st.text_input(
        "粘贴已选择的文本",
        value="",
        key="pasted_anno_text",
        help="将从文档中选择并复制的文本粘贴到这里"
    )

    # 自动查找文本位置
    if pasted_text and pasted_text in doc_text:
        # 查找所有匹配的位置
        start_pos = doc_text.index(pasted_text)
        end_pos = start_pos + len(pasted_text)
        st.session_state.anno_selected_text = pasted_text
        st.session_state.anno_selected_start = start_pos
        st.session_state.anno_selected_end = end_pos

    # 显示当前选择状态
    selected_text = st.session_state.get('anno_selected_text', '')
    selected_start = st.session_state.get('anno_selected_start', -1)
    selected_end = st.session_state.get('anno_selected_end', -1)

    if selected_text and selected_start >= 0 and selected_end > selected_start:
        st.success(f"✅ 已识别文本: 「{selected_text[:80]}{'...' if len(selected_text) > 80 else ''}」")
        st.caption(f"在原文中的位置: {selected_start} - {selected_end}，长度: {len(selected_text)} 字符")
    else:
        st.warning("👆 请在上方文本中选择内容并粘贴到此处")

    col1, col2 = st.columns([1, 1])
    with col1:
        code_names = [c.name for c in codes]
        selected_codes = st.multiselect("选择编码", code_names, key="anno_codes")

    with col2:
        note = st.text_input("备注（可选）", key="anno_note")

    # 备用的手动定位方式
    with st.expander("📝 备选：手动指定位置"):
        st.info("如果自动识别失败，可使用此方式")
        text_len = len(doc_text)
        m_start = st.number_input("起始位置", min_value=0, max_value=max(0, text_len-1), value=0, key="manual_start")
        m_end = st.number_input("结束位置", min_value=1, max_value=max(1, text_len), value=min(50, text_len), key="manual_end")

        if m_start < m_end:
            manual_preview = doc_text[m_start:m_end]
            st.text(f"预览: {manual_preview[:100]}...")

            if st.button("使用此位置", key="use_manual_pos"):
                st.session_state.anno_selected_text = manual_preview
                st.session_state.anno_selected_start = m_start
                st.session_state.anno_selected_end = m_end
                st.rerun()

    # 使用选择的位置
    final_start = selected_start if selected_start >= 0 else -1
    final_end = selected_end if selected_end > selected_start else -1
    final_text = st.session_state.anno_selected_text if final_start >= 0 else ""

    if st.button("➕ 添加标注", key="add_annotation", type="primary"):
        if not selected_codes:
            st.warning("请选择至少一个编码")
        elif final_start < 0 or not final_text:
            st.warning("请先选择要标注的文本")
        else:
            segment = coder.add_segment(
                doc_id=doc_id,
                start=final_start,
                end=final_end,
                text=final_text,
                codes=selected_codes,
                note=note
            )
            if segment:
                log_message(f"已添加标注: {final_text[:20]}...")
                st.success("标注添加成功！")
                # 清除选择
                st.session_state.anno_selected_text = ""
                st.session_state.anno_selected_start = -1
                st.session_state.anno_selected_end = -1
                st.rerun()
            else:
                st.error("添加失败")
    
    st.divider()
    
    # 显示该文档的已有标注
    st.markdown("**该文档的标注**")
    if doc_segments:
        seg_data = []
        for i, seg in enumerate(doc_segments):
            seg_data.append({
                "序号": i + 1,
                "文本片段": seg.text[:50] + ("..." if len(seg.text) > 50 else ""),
                "编码": ", ".join(seg.codes),
                "位置": f"{seg.start_pos}-{seg.end_pos}",
                "备注": seg.note or "-"
            })
        
        df = pd.DataFrame(seg_data)
        st.dataframe(df, width='stretch', hide_index=True)
        
        # 删除标注
        col1, col2 = st.columns([3, 1])
        with col1:
            # 找到该文档片段在总列表中的索引
            doc_seg_indices = [i for i, seg in enumerate(coder.segments) if seg.document_id == doc_id]
            if doc_seg_indices:
                delete_options = [f"{i+1}. {coder.segments[idx].text[:30]}..." for i, idx in enumerate(doc_seg_indices)]
                delete_select = st.selectbox("选择要删除的标注", delete_options, key="delete_anno_select")
        with col2:
            st.write("")
            st.write("")
            if st.button("删除标注", key="delete_annotation", type="secondary"):
                if doc_seg_indices:
                    selected_idx = delete_options.index(delete_select)
                    actual_idx = doc_seg_indices[selected_idx]
                    if coder.remove_segment(actual_idx):
                        log_message("已删除标注", level="warning")
                        st.rerun()
    else:
        st.info("该文档暂无标注")



def _generate_highlighted_html(text: str, segments: List[CodedSegment], scheme: CodingScheme) -> str:
    """
    生成带高亮的HTML文本
    
    Args:
        text: 原始文本
        segments: 已编码片段列表
        scheme: 编码方案
        
    Returns:
        str: 带高亮标记的HTML字符串
    """
    import html
    
    if not segments:
        return html.escape(text)
    
    # 按起始位置排序片段
    sorted_segments = sorted(segments, key=lambda s: s.start_pos)
    
    # 构建高亮HTML
    result = []
    last_end = 0
    
    for seg in sorted_segments:
        # 添加未高亮的部分
        if seg.start_pos > last_end:
            result.append(html.escape(text[last_end:seg.start_pos]))
        
        # 获取编码颜色（使用第一个编码的颜色）
        color = "#ffeb3b"  # 默认黄色
        if seg.codes:
            code = scheme.get_code(seg.codes[0])
            if code:
                color = code.color
        
        # 添加高亮部分
        codes_str = ", ".join(seg.codes)
        highlighted_text = html.escape(text[seg.start_pos:seg.end_pos])
        result.append(
            f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;" '
            f'title="编码: {codes_str}">{highlighted_text}</span>'
        )
        
        last_end = seg.end_pos
    
    # 添加剩余部分
    if last_end < len(text):
        result.append(html.escape(text[last_end:]))
    
    return ''.join(result)


def _render_statistics_tab(coder: QualitativeCoder):
    """渲染编码统计标签页"""
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    from utils.session_state import log_message
    
    st.subheader("编码统计")
    
    if coder.get_segment_count() == 0:
        st.info("暂无编码数据，请先进行文本标注")
        return
    
    # 获取统计数据
    stats = coder.get_code_statistics()
    doc_stats = coder.get_document_statistics()
    
    # 总体统计
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("编码总数", coder.scheme.get_code_count())
    with col2:
        st.metric("标注片段数", coder.get_segment_count())
    with col3:
        st.metric("已编码文档数", len(coder.get_coded_documents()))
    with col4:
        total_coverage = sum(s.text_coverage for s in stats.values())
        st.metric("覆盖文本量", f"{total_coverage} 字符")
    
    st.divider()
    
    # 编码使用频次统计
    st.markdown("**编码使用频次**")
    
    if stats:
        # 准备数据
        freq_data = []
        for code_name, code_stats in stats.items():
            code = coder.scheme.get_code(code_name)
            freq_data.append({
                "编码": code_name,
                "使用频次": code_stats.frequency,
                "覆盖文本量": code_stats.text_coverage,
                "涉及文档数": code_stats.document_count,
                "颜色": code.color if code else "#3498db"
            })
        
        df = pd.DataFrame(freq_data)
        df = df.sort_values("使用频次", ascending=False)
        
        # 显示表格
        st.dataframe(df[["编码", "使用频次", "覆盖文本量", "涉及文档数"]], 
                     width='stretch', hide_index=True)
        
        # 绘制柱状图
        if len(freq_data) > 0:
            fig = px.bar(
                df, 
                x="编码", 
                y="使用频次",
                color="编码",
                title="编码使用频次分布"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    # 文档编码统计
    st.markdown("**文档编码统计**")
    
    if doc_stats:
        doc_data = []
        for doc_id, doc_stat in doc_stats.items():
            doc_data.append({
                "文档": doc_id,
                "标注片段数": doc_stat['segment_count'],
                "使用编码数": len(doc_stat['codes_used']),
                "覆盖文本量": doc_stat['text_coverage']
            })
        
        doc_df = pd.DataFrame(doc_data)
        st.dataframe(doc_df, width='stretch', hide_index=True)
    
    st.divider()
    
    # 导出统计
    st.markdown("**导出数据**")
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = coder.export_to_csv()
        st.download_button(
            label="📥 导出编码结果 (CSV)",
            data=csv_data,
            file_name="coding_results.csv",
            mime="text/csv",
            key="export_coding_csv"
        )
    
    with col2:
        stats_csv = coder.export_statistics_csv()
        st.download_button(
            label="📥 导出统计数据 (CSV)",
            data=stats_csv,
            file_name="coding_statistics.csv",
            mime="text/csv",
            key="export_stats_csv"
        )



def _render_save_load_tab(coder: QualitativeCoder):
    """渲染保存/加载标签页"""
    import streamlit as st
    from utils.session_state import log_message
    
    st.subheader("保存/加载编码方案")
    
    col1, col2 = st.columns(2)
    
    # 保存方案
    with col1:
        st.markdown("**保存编码方案**")
        
        # 显示当前方案信息
        st.info(f"当前方案: {coder.scheme.name}\n"
                f"编码数: {coder.scheme.get_code_count()}\n"
                f"标注片段数: {coder.get_segment_count()}")
        
        # 导出为JSON下载
        json_data = coder.to_dict()
        json_str = __import__('json').dumps(json_data, ensure_ascii=False, indent=2)
        
        st.download_button(
            label="💾 下载编码方案 (JSON)",
            data=json_str,
            file_name=f"{coder.scheme.name}.json",
            mime="application/json",
            key="download_scheme"
        )
    
    # 加载方案
    with col2:
        st.markdown("**加载编码方案**")
        
        uploaded_file = st.file_uploader(
            "上传编码方案文件 (JSON)",
            type=["json"],
            key="upload_scheme"
        )
        
        if uploaded_file:
            if st.button("加载方案", key="load_scheme_btn"):
                try:
                    content = uploaded_file.read().decode('utf-8')
                    data = __import__('json').loads(content)
                    
                    # 创建新的编码器
                    new_coder = QualitativeCoder.from_dict(data)
                    
                    # 更新会话状态
                    st.session_state["coding_scheme"] = new_coder
                    
                    log_message(f"已加载编码方案: {new_coder.scheme.name}")
                    st.success(f"成功加载编码方案: {new_coder.scheme.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"加载失败: {str(e)}")
    
    st.divider()
    
    # 重置方案
    st.markdown("**重置编码方案**")
    st.warning("⚠️ 重置将清除所有编码和标注数据，此操作不可撤销！")
    
    if st.button("🗑️ 重置编码方案", key="reset_scheme", type="secondary"):
        st.session_state["coding_scheme"] = QualitativeCoder()
        log_message("已重置编码方案", level="warning")
        st.rerun()
