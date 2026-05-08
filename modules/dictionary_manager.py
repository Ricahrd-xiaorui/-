# -*- coding: utf-8 -*-
"""
专业词典管理模块 (Professional Dictionary Management Module)

本模块提供专业词典的管理功能，包括：
- 词典的创建、导入、导出
- 词汇的增删改查
- 与jieba分词器的集成
- 术语在文本中的识别和频率统计

Requirements: 9.1, 9.3, 9.4, 9.5, 9.6, 9.9, 9.10
"""

import os
import json
import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import Counter
import jieba


@dataclass
class Dictionary:
    """
    词典类 - 存储词汇及其词性标注
    
    Attributes:
        name: 词典名称
        words: 词汇字典，键为词汇，值为词性标注（可选）
        description: 词典描述
    """
    name: str
    words: Dict[str, Optional[str]] = field(default_factory=dict)
    description: str = ""
    
    def add(self, word: str, pos: Optional[str] = None) -> bool:
        """
        添加词汇到词典
        
        Args:
            word: 要添加的词汇
            pos: 词性标注（可选）
            
        Returns:
            bool: 是否成功添加（词汇已存在则返回False）
        """
        word = word.strip()
        if not word:
            return False
        if word in self.words:
            # 更新词性
            self.words[word] = pos
            return True
        self.words[word] = pos
        return True
    
    def remove(self, word: str) -> bool:
        """
        从词典中删除词汇
        
        Args:
            word: 要删除的词汇
            
        Returns:
            bool: 是否成功删除
        """
        word = word.strip()
        if word in self.words:
            del self.words[word]
            return True
        return False
    
    def contains(self, word: str) -> bool:
        """
        检查词典是否包含指定词汇
        
        Args:
            word: 要检查的词汇
            
        Returns:
            bool: 是否包含该词汇
        """
        return word.strip() in self.words
    
    def get_pos(self, word: str) -> Optional[str]:
        """
        获取词汇的词性标注
        
        Args:
            word: 词汇
            
        Returns:
            Optional[str]: 词性标注，如果不存在则返回None
        """
        return self.words.get(word.strip())
    
    def update_pos(self, word: str, pos: Optional[str]) -> bool:
        """
        更新词汇的词性标注
        
        Args:
            word: 词汇
            pos: 新的词性标注
            
        Returns:
            bool: 是否成功更新
        """
        word = word.strip()
        if word in self.words:
            self.words[word] = pos
            return True
        return False
    
    def to_list(self) -> List[Tuple[str, Optional[str]]]:
        """
        将词典转换为列表格式
        
        Returns:
            List[Tuple[str, Optional[str]]]: (词汇, 词性) 元组列表
        """
        return [(word, pos) for word, pos in sorted(self.words.items())]
    
    def get_words(self) -> Set[str]:
        """
        获取所有词汇集合
        
        Returns:
            Set[str]: 词汇集合
        """
        return set(self.words.keys())
    
    def size(self) -> int:
        """
        获取词典大小
        
        Returns:
            int: 词汇数量
        """
        return len(self.words)
    
    def to_dict(self) -> dict:
        """
        将词典转换为字典格式（用于序列化）
        
        Returns:
            dict: 词典数据
        """
        return {
            'name': self.name,
            'words': self.words,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Dictionary':
        """
        从字典数据创建词典实例
        
        Args:
            data: 词典数据
            
        Returns:
            Dictionary: 词典实例
        """
        return cls(
            name=data.get('name', ''),
            words=data.get('words', {}),
            description=data.get('description', '')
        )


class DictionaryManager:
    """
    专业词典管理器 - 管理多个词典并与jieba分词器集成
    
    Attributes:
        dictionaries: 词典字典，键为词典名称
        active_dictionaries: 激活的词典名称列表
    """
    
    def __init__(self):
        """初始化词典管理器"""
        self.dictionaries: Dict[str, Dictionary] = {}
        self.active_dictionaries: List[str] = []
        self._jieba_applied: bool = False
    
    def create_dictionary(self, name: str, description: str = "") -> Optional[Dictionary]:
        """
        创建新词典
        
        Args:
            name: 词典名称
            description: 词典描述
            
        Returns:
            Optional[Dictionary]: 创建的词典，如果名称已存在则返回None
        """
        name = name.strip()
        if not name or name in self.dictionaries:
            return None
        
        dictionary = Dictionary(name=name, description=description)
        self.dictionaries[name] = dictionary
        return dictionary

    def import_from_file(self, name: str, file_path: str, description: str = "") -> Tuple[bool, str]:
        """
        从文本文件导入词典

        Args:
            name: 词典名称
            file_path: 词典文件路径（每行一个词汇）
            description: 词典描述

        Returns:
            Tuple[bool, str]: (是否成功, 消息)
        """
        if not os.path.exists(file_path):
            return False, f"文件不存在: {file_path}"

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                words = [line.strip() for line in f if line.strip()]

            dictionary = self.create_dictionary(name, description)
            if dictionary is None:
                dictionary = self.dictionaries.get(name)
                if dictionary:
                    dictionary.description = description

            added_count = 0
            for word in words:
                if dictionary.add(word):
                    added_count += 1

            if name not in self.active_dictionaries:
                self.activate_dictionary(name)

            return True, f"成功导入 {added_count} 个词汇"
        except Exception as e:
            return False, f"导入失败: {str(e)}"

    def get_dictionary(self, name: str) -> Optional[Dictionary]:
        """
        获取指定名称的词典
        
        Args:
            name: 词典名称
            
        Returns:
            Optional[Dictionary]: 词典实例，不存在则返回None
        """
        return self.dictionaries.get(name)
    
    def remove_dictionary(self, name: str) -> bool:
        """
        删除词典
        
        Args:
            name: 词典名称
            
        Returns:
            bool: 是否成功删除
        """
        if name in self.dictionaries:
            del self.dictionaries[name]
            if name in self.active_dictionaries:
                self.active_dictionaries.remove(name)
            return True
        return False
    
    def list_dictionaries(self) -> List[str]:
        """
        列出所有词典名称
        
        Returns:
            List[str]: 词典名称列表
        """
        return list(self.dictionaries.keys())
    
    def import_dictionary(self, filepath: str, name: str, 
                         description: str = "") -> bool:
        """
        从TXT文件导入词典（每行一个词，可选词性用空格或制表符分隔）
        
        Args:
            filepath: 文件路径
            name: 词典名称
            description: 词典描述
            
        Returns:
            bool: 是否成功导入
        """
        try:
            # 尝试不同编码读取文件
            content = None
            for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030']:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                return False
            
            return self.import_dictionary_from_text(content, name, description)
            
        except Exception:
            return False
    
    def import_dictionary_from_text(self, content: str, name: str,
                                    description: str = "") -> bool:
        """
        从文本内容导入词典
        
        Args:
            content: 文本内容（每行一个词，可选词性用空格或制表符分隔）
            name: 词典名称
            description: 词典描述
            
        Returns:
            bool: 是否成功导入
        """
        try:
            name = name.strip()
            if not name:
                return False
            
            # 创建或获取词典
            if name in self.dictionaries:
                dictionary = self.dictionaries[name]
            else:
                dictionary = Dictionary(name=name, description=description)
                self.dictionaries[name] = dictionary
            
            # 解析内容
            lines = content.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 尝试解析词汇和词性
                parts = re.split(r'[\s\t]+', line, maxsplit=1)
                word = parts[0].strip()
                pos = parts[1].strip() if len(parts) > 1 else None
                
                if word:
                    dictionary.add(word, pos)
            
            return True
            
        except Exception:
            return False
    
    def export_dictionary(self, name: str, filepath: str) -> bool:
        """
        导出词典到TXT文件
        
        Args:
            name: 词典名称
            filepath: 导出文件路径
            
        Returns:
            bool: 是否成功导出
        """
        try:
            dictionary = self.dictionaries.get(name)
            if not dictionary:
                return False
            
            content = self.export_dictionary_to_text(name)
            if content is None:
                return False
            
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception:
            return False
    
    def export_dictionary_to_text(self, name: str) -> Optional[str]:
        """
        将词典导出为文本格式
        
        Args:
            name: 词典名称
            
        Returns:
            Optional[str]: 文本内容，失败返回None
        """
        dictionary = self.dictionaries.get(name)
        if not dictionary:
            return None
        
        lines = []
        for word, pos in sorted(dictionary.words.items()):
            if pos:
                lines.append(f"{word}\t{pos}")
            else:
                lines.append(word)
        
        return '\n'.join(lines)
    
    def add_word(self, dict_name: str, word: str, pos: Optional[str] = None) -> bool:
        """
        向指定词典添加词汇
        
        Args:
            dict_name: 词典名称
            word: 词汇
            pos: 词性标注
            
        Returns:
            bool: 是否成功添加
        """
        dictionary = self.dictionaries.get(dict_name)
        if dictionary:
            result = dictionary.add(word, pos)
            if result and dict_name in self.active_dictionaries:
                # 如果词典已激活，同步更新jieba
                self._add_word_to_jieba(word, pos)
            return result
        return False
    
    def remove_word(self, dict_name: str, word: str) -> bool:
        """
        从指定词典删除词汇
        
        Args:
            dict_name: 词典名称
            word: 词汇
            
        Returns:
            bool: 是否成功删除
        """
        dictionary = self.dictionaries.get(dict_name)
        if dictionary:
            return dictionary.remove(word)
        return False
    
    def activate_dictionary(self, name: str) -> bool:
        """
        激活词典（将词典词汇加入jieba分词）
        
        Args:
            name: 词典名称
            
        Returns:
            bool: 是否成功激活
        """
        if name not in self.dictionaries:
            return False
        
        if name not in self.active_dictionaries:
            self.active_dictionaries.append(name)
        
        # 应用到jieba
        self.apply_to_jieba()
        return True
    
    def deactivate_dictionary(self, name: str) -> bool:
        """
        停用词典
        
        Args:
            name: 词典名称
            
        Returns:
            bool: 是否成功停用
        """
        if name in self.active_dictionaries:
            self.active_dictionaries.remove(name)
            # 重新应用剩余激活的词典
            self.apply_to_jieba()
            return True
        return False
    
    def is_active(self, name: str) -> bool:
        """
        检查词典是否已激活
        
        Args:
            name: 词典名称
            
        Returns:
            bool: 是否已激活
        """
        return name in self.active_dictionaries
    
    def apply_to_jieba(self) -> None:
        """
        将所有激活的词典词汇应用到jieba分词器
        """
        # 收集所有激活词典的词汇
        for dict_name in self.active_dictionaries:
            dictionary = self.dictionaries.get(dict_name)
            if dictionary:
                for word, pos in dictionary.words.items():
                    self._add_word_to_jieba(word, pos)
        
        self._jieba_applied = True
    
    def _add_word_to_jieba(self, word: str, pos: Optional[str] = None) -> None:
        """
        将单个词汇添加到jieba用户词典
        
        Args:
            word: 词汇
            pos: 词性标注
        """
        if pos:
            jieba.add_word(word, tag=pos)
        else:
            jieba.add_word(word)
    
    def get_all_active_words(self) -> Set[str]:
        """
        获取所有激活词典中的词汇
        
        Returns:
            Set[str]: 词汇集合
        """
        words = set()
        for dict_name in self.active_dictionaries:
            dictionary = self.dictionaries.get(dict_name)
            if dictionary:
                words.update(dictionary.get_words())
        return words
    
    def find_terms_in_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        在文本中查找所有激活词典中的术语
        
        Args:
            text: 要搜索的文本
            
        Returns:
            List[Tuple[str, int, int]]: (术语, 起始位置, 结束位置) 列表
        """
        results = []
        active_words = self.get_all_active_words()
        
        # 按词汇长度降序排序，优先匹配长词
        sorted_words = sorted(active_words, key=len, reverse=True)
        
        # 记录已匹配的位置，避免重叠
        matched_positions = set()
        
        for word in sorted_words:
            if not word:
                continue
            
            # 使用正则表达式查找所有匹配
            pattern = re.escape(word)
            for match in re.finditer(pattern, text):
                start, end = match.start(), match.end()
                
                # 检查是否与已匹配位置重叠
                positions = set(range(start, end))
                if not positions & matched_positions:
                    results.append((word, start, end))
                    matched_positions.update(positions)
        
        # 按起始位置排序
        results.sort(key=lambda x: x[1])
        return results
    
    def count_term_frequency(self, texts: List[str]) -> Dict[str, int]:
        """
        统计术语在文本集合中的出现频率
        
        Args:
            texts: 文本列表
            
        Returns:
            Dict[str, int]: 术语频率字典
        """
        frequency = Counter()
        active_words = self.get_all_active_words()
        
        for text in texts:
            for word in active_words:
                if word:
                    # 计算词汇在文本中出现的次数
                    count = len(re.findall(re.escape(word), text))
                    if count > 0:
                        frequency[word] += count
        
        return dict(frequency)
    
    def save_all(self, filepath: str) -> bool:
        """
        保存所有词典到JSON文件
        
        Args:
            filepath: 保存路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            data = {
                'dictionaries': {
                    name: dictionary.to_dict() 
                    for name, dictionary in self.dictionaries.items()
                },
                'active_dictionaries': self.active_dictionaries
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
    
    def load_all(self, filepath: str) -> bool:
        """
        从JSON文件加载所有词典
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载词典
            self.dictionaries = {}
            for name, dict_data in data.get('dictionaries', {}).items():
                self.dictionaries[name] = Dictionary.from_dict(dict_data)
            
            # 加载激活状态
            self.active_dictionaries = data.get('active_dictionaries', [])
            
            # 验证激活的词典是否存在
            self.active_dictionaries = [
                name for name in self.active_dictionaries 
                if name in self.dictionaries
            ]
            
            # 应用到jieba
            if self.active_dictionaries:
                self.apply_to_jieba()
            
            return True
            
        except Exception:
            return False
    
    def to_dict(self) -> dict:
        """
        将管理器状态转换为字典（用于序列化）
        
        Returns:
            dict: 管理器状态数据
        """
        return {
            'dictionaries': {
                name: dictionary.to_dict() 
                for name, dictionary in self.dictionaries.items()
            },
            'active_dictionaries': self.active_dictionaries
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DictionaryManager':
        """
        从字典数据创建管理器实例
        
        Args:
            data: 管理器状态数据
            
        Returns:
            DictionaryManager: 管理器实例
        """
        manager = cls()
        
        for name, dict_data in data.get('dictionaries', {}).items():
            manager.dictionaries[name] = Dictionary.from_dict(dict_data)
        
        manager.active_dictionaries = data.get('active_dictionaries', [])
        
        # 验证激活的词典是否存在
        manager.active_dictionaries = [
            name for name in manager.active_dictionaries 
            if name in manager.dictionaries
        ]
        
        return manager



# ============================================================================
# Streamlit UI 渲染函数
# ============================================================================

def render_dictionary_manager():
    """
    渲染专业词典管理模块UI
    
    Requirements: 9.2, 9.7, 9.8, 9.9, 9.10
    """
    import streamlit as st
    import pandas as pd
    from utils.session_state import log_message
    
    st.header("专业词典管理")
    
    # 功能介绍
    with st.expander("📖 功能介绍", expanded=False):
        st.markdown("""
        **专业词典管理模块** 用于管理和使用专业领域词典，提高分词准确性。
        
        **主要功能：**
        - 📥 **词典导入**：支持导入TXT格式的自定义专业词典
        - ✏️ **在线编辑**：添加、删除、修改词典中的词条
        - 🎯 **术语高亮**：在文本中高亮显示匹配的专业术语
        - 📊 **频率统计**：统计专业术语在文本中的出现频率
        - 💾 **保存加载**：支持词典的保存和加载以便复用
        - 📤 **词典导出**：支持导出词典为TXT文件
        
        **词典格式：**
        - TXT文件，每行一个词
        - 可选：词汇后用空格或制表符分隔词性标注
        - 示例：`人工智能 n` 或 `人工智能`
        """)
    
    # 初始化词典管理器
    if st.session_state.get("dictionary_manager") is None:
        st.session_state["dictionary_manager"] = DictionaryManager()
    
    manager: DictionaryManager = st.session_state["dictionary_manager"]
    
    # 创建标签页
    tabs = st.tabs(["词典管理", "在线编辑", "术语高亮", "频率统计", "保存/加载"])
    
    # ========== 词典管理标签页 ==========
    with tabs[0]:
        st.subheader("词典列表")
        
        # 显示现有词典
        dict_names = manager.list_dictionaries()
        if dict_names:
            dict_data = []
            for name in dict_names:
                dictionary = manager.get_dictionary(name)
                if dictionary:
                    dict_data.append({
                        "词典名称": name,
                        "词汇数量": dictionary.size(),
                        "状态": "✅ 已激活" if manager.is_active(name) else "⬜ 未激活",
                        "描述": dictionary.description or "-"
                    })
            
            df = pd.DataFrame(dict_data)
            st.dataframe(df, width='stretch', hide_index=True)
            
            # 词典操作
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_dict = st.selectbox(
                    "选择词典",
                    dict_names,
                    key="dict_select_manage"
                )
            
            with col2:
                if selected_dict:
                    if manager.is_active(selected_dict):
                        if st.button("停用词典", key="deactivate_dict"):
                            manager.deactivate_dictionary(selected_dict)
                            log_message(f"已停用词典: {selected_dict}")
                            st.rerun()
                    else:
                        if st.button("激活词典", key="activate_dict"):
                            manager.activate_dictionary(selected_dict)
                            log_message(f"已激活词典: {selected_dict}")
                            st.rerun()
            
            with col3:
                if selected_dict:
                    if st.button("删除词典", key="delete_dict", type="secondary"):
                        manager.remove_dictionary(selected_dict)
                        log_message(f"已删除词典: {selected_dict}", level="warning")
                        st.rerun()
        else:
            st.info("暂无词典，请创建或导入词典")
        
        st.divider()
        
        # 创建新词典
        st.subheader("创建新词典")
        col1, col2 = st.columns(2)
        with col1:
            new_dict_name = st.text_input("词典名称", key="new_dict_name")
        with col2:
            new_dict_desc = st.text_input("词典描述（可选）", key="new_dict_desc")
        
        if st.button("创建词典", key="create_dict"):
            if new_dict_name:
                result = manager.create_dictionary(new_dict_name, new_dict_desc)
                if result:
                    log_message(f"已创建词典: {new_dict_name}")
                    st.success(f"成功创建词典: {new_dict_name}")
                    st.rerun()
                else:
                    st.error("创建失败，词典名称可能已存在")
            else:
                st.warning("请输入词典名称")
        
        st.divider()
        
        # 导入词典
        st.subheader("导入词典")
        uploaded_file = st.file_uploader(
            "上传词典文件（TXT格式，每行一个词）",
            type=["txt"],
            key="dict_file_uploader"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            import_dict_name = st.text_input("导入后的词典名称", key="import_dict_name")
        with col2:
            import_dict_desc = st.text_input("词典描述（可选）", key="import_dict_desc")
        
        if st.button("导入词典", key="import_dict"):
            if uploaded_file and import_dict_name:
                content = uploaded_file.read().decode('utf-8', errors='replace')
                if manager.import_dictionary_from_text(content, import_dict_name, import_dict_desc):
                    dictionary = manager.get_dictionary(import_dict_name)
                    word_count = dictionary.size() if dictionary else 0
                    log_message(f"已导入词典: {import_dict_name}，共 {word_count} 个词")
                    st.success(f"成功导入词典: {import_dict_name}，共 {word_count} 个词")
                    st.rerun()
                else:
                    st.error("导入失败")
            else:
                st.warning("请上传文件并输入词典名称")
    
    # ========== 在线编辑标签页 ==========
    with tabs[1]:
        st.subheader("在线编辑词典")
        
        dict_names = manager.list_dictionaries()
        if not dict_names:
            st.info("请先创建或导入词典")
        else:
            selected_dict = st.selectbox(
                "选择要编辑的词典",
                dict_names,
                key="dict_select_edit"
            )
            
            if selected_dict:
                dictionary = manager.get_dictionary(selected_dict)
                
                # 添加词汇
                st.markdown("**添加词汇**")
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    new_word = st.text_input("词汇", key="new_word_input")
                with col2:
                    new_pos = st.text_input("词性（可选）", key="new_pos_input")
                with col3:
                    st.write("")  # 占位
                    st.write("")  # 占位
                    if st.button("添加", key="add_word_btn"):
                        if new_word:
                            if manager.add_word(selected_dict, new_word, new_pos if new_pos else None):
                                log_message(f"已添加词汇: {new_word}")
                                st.success(f"已添加: {new_word}")
                                st.rerun()
                            else:
                                st.error("添加失败")
                        else:
                            st.warning("请输入词汇")
                
                # 批量添加
                st.markdown("**批量添加词汇**")
                batch_words = st.text_area(
                    "输入词汇（每行一个，可选词性用空格分隔）",
                    height=150,
                    key="batch_words_input"
                )
                if st.button("批量添加", key="batch_add_btn"):
                    if batch_words:
                        lines = batch_words.strip().split('\n')
                        added_count = 0
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split(maxsplit=1)
                            word = parts[0]
                            pos = parts[1] if len(parts) > 1 else None
                            if manager.add_word(selected_dict, word, pos):
                                added_count += 1
                        log_message(f"批量添加了 {added_count} 个词汇")
                        st.success(f"成功添加 {added_count} 个词汇")
                        st.rerun()
                
                st.divider()
                
                # 显示和编辑现有词汇
                st.markdown("**现有词汇**")
                if dictionary and dictionary.size() > 0:
                    words_list = dictionary.to_list()
                    
                    # 搜索过滤
                    search_term = st.text_input("搜索词汇", key="search_word_input")
                    if search_term:
                        words_list = [(w, p) for w, p in words_list if search_term in w]
                    
                    # 分页显示
                    page_size = 50
                    total_pages = (len(words_list) + page_size - 1) // page_size
                    
                    if total_pages > 1:
                        page = st.number_input(
                            f"页码 (共 {total_pages} 页)",
                            min_value=1,
                            max_value=total_pages,
                            value=1,
                            key="word_page"
                        )
                    else:
                        page = 1
                    
                    start_idx = (page - 1) * page_size
                    end_idx = min(start_idx + page_size, len(words_list))
                    
                    # 显示词汇表格
                    df = pd.DataFrame(
                        words_list[start_idx:end_idx],
                        columns=["词汇", "词性"]
                    )
                    df["词性"] = df["词性"].fillna("-")
                    st.dataframe(df, width='stretch', hide_index=True)
                    
                    st.write(f"显示 {start_idx + 1}-{end_idx} / 共 {len(words_list)} 个词汇")
                    
                    # 删除词汇
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        word_to_delete = st.text_input("输入要删除的词汇", key="delete_word_input")
                    with col2:
                        st.write("")
                        st.write("")
                        if st.button("删除", key="delete_word_btn", type="secondary"):
                            if word_to_delete:
                                if manager.remove_word(selected_dict, word_to_delete):
                                    log_message(f"已删除词汇: {word_to_delete}")
                                    st.success(f"已删除: {word_to_delete}")
                                    st.rerun()
                                else:
                                    st.error("删除失败，词汇可能不存在")
                else:
                    st.info("词典为空")
    
    # ========== 术语高亮标签页 ==========
    with tabs[2]:
        st.subheader("术语高亮")
        
        # 检查是否有激活的词典
        if not manager.active_dictionaries:
            st.warning("请先激活至少一个词典")
        else:
            st.info(f"当前激活的词典: {', '.join(manager.active_dictionaries)}")
            
            # 输入文本
            input_text = st.text_area(
                "输入要分析的文本",
                height=200,
                key="highlight_text_input"
            )
            
            # 或从已加载的文件中选择
            if st.session_state.get("raw_texts"):
                st.markdown("**或从已加载的文件中选择：**")
                file_names = st.session_state.get("file_names", [])
                selected_file = st.selectbox(
                    "选择文件",
                    [""] + file_names,
                    key="highlight_file_select"
                )
                if selected_file:
                    file_idx = file_names.index(selected_file)
                    input_text = st.session_state["raw_texts"][file_idx]
            
            if st.button("分析术语", key="analyze_terms_btn"):
                if input_text:
                    # 查找术语
                    terms = manager.find_terms_in_text(input_text)
                    
                    if terms:
                        st.success(f"找到 {len(terms)} 个术语匹配")
                        
                        # 高亮显示
                        highlighted_text = input_text
                        # 从后向前替换，避免位置偏移
                        for term, start, end in reversed(terms):
                            highlighted_text = (
                                highlighted_text[:start] + 
                                f"**:red[{term}]**" + 
                                highlighted_text[end:]
                            )
                        
                        st.markdown("**高亮结果：**")
                        st.markdown(highlighted_text)
                        
                        # 显示术语列表
                        st.markdown("**匹配的术语：**")
                        term_list = list(set(t[0] for t in terms))
                        term_counts = {}
                        for t in terms:
                            term_counts[t[0]] = term_counts.get(t[0], 0) + 1
                        
                        df = pd.DataFrame([
                            {"术语": term, "出现次数": term_counts[term]}
                            for term in sorted(term_list)
                        ])
                        st.dataframe(df, width='stretch', hide_index=True)
                    else:
                        st.info("未找到匹配的术语")
                else:
                    st.warning("请输入文本")
    
    # ========== 频率统计标签页 ==========
    with tabs[3]:
        st.subheader("术语频率统计")
        
        # 检查是否有激活的词典
        if not manager.active_dictionaries:
            st.warning("请先激活至少一个词典")
        elif not st.session_state.get("raw_texts"):
            st.warning("请先在数据加载模块中加载文本文件")
        else:
            st.info(f"当前激活的词典: {', '.join(manager.active_dictionaries)}")
            st.info(f"已加载 {len(st.session_state['raw_texts'])} 个文本文件")
            
            if st.button("统计术语频率", key="count_freq_btn"):
                with st.spinner("正在统计..."):
                    # 统计频率
                    frequency = manager.count_term_frequency(st.session_state["raw_texts"])
                    
                    if frequency:
                        # 保存到会话状态
                        st.session_state["term_frequencies"] = frequency
                        
                        # 显示结果
                        st.success(f"统计完成，共 {len(frequency)} 个术语")
                        
                        # 排序并显示
                        sorted_freq = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
                        df = pd.DataFrame(sorted_freq, columns=["术语", "频率"])
                        
                        # 显示统计信息
                        col1, col2, col3 = st.columns(3)
                        col1.metric("术语总数", len(frequency))
                        col2.metric("总出现次数", sum(frequency.values()))
                        col3.metric("平均频率", f"{sum(frequency.values()) / len(frequency):.1f}")
                        
                        st.dataframe(df, width='stretch', hide_index=True)
                        
                        # 导出按钮
                        csv = df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="导出为CSV",
                            data=csv,
                            file_name="term_frequency.csv",
                            mime="text/csv",
                            key="export_freq_csv"
                        )
                        
                        log_message(f"术语频率统计完成，共 {len(frequency)} 个术语")
                    else:
                        st.info("未找到匹配的术语")
            
            # 显示之前的统计结果
            if st.session_state.get("term_frequencies"):
                st.markdown("**上次统计结果：**")
                frequency = st.session_state["term_frequencies"]
                sorted_freq = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
                df = pd.DataFrame(sorted_freq, columns=["术语", "频率"])
                st.dataframe(df, width='stretch', hide_index=True, height=300)
    
    # ========== 保存/加载标签页 ==========
    with tabs[4]:
        st.subheader("保存和加载词典")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**保存所有词典**")
            save_path = st.text_input(
                "保存路径",
                value="data/dictionaries.json",
                key="save_dict_path"
            )
            if st.button("保存词典", key="save_all_dict"):
                if manager.save_all(save_path):
                    log_message(f"词典已保存到: {save_path}")
                    st.success(f"成功保存到: {save_path}")
                else:
                    st.error("保存失败")
            
            st.divider()
            
            # 导出单个词典
            st.markdown("**导出单个词典为TXT**")
            dict_names = manager.list_dictionaries()
            if dict_names:
                export_dict = st.selectbox(
                    "选择要导出的词典",
                    dict_names,
                    key="export_dict_select"
                )
                if export_dict:
                    content = manager.export_dictionary_to_text(export_dict)
                    if content:
                        st.download_button(
                            label="下载词典文件",
                            data=content,
                            file_name=f"{export_dict}.txt",
                            mime="text/plain",
                            key="download_dict_txt"
                        )
        
        with col2:
            st.markdown("**加载词典**")
            load_file = st.file_uploader(
                "上传词典配置文件（JSON格式）",
                type=["json"],
                key="load_dict_file"
            )
            if st.button("加载词典", key="load_all_dict"):
                if load_file:
                    # 保存临时文件
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
                        tmp.write(load_file.read())
                        tmp_path = tmp.name
                    
                    if manager.load_all(tmp_path):
                        st.session_state["dictionary_manager"] = manager
                        log_message("词典加载成功")
                        st.success("词典加载成功")
                        st.rerun()
                    else:
                        st.error("加载失败，请检查文件格式")
                    
                    # 清理临时文件
                    os.unlink(tmp_path)
                else:
                    st.warning("请上传文件")
            
            st.divider()
            
            # 显示当前状态
            st.markdown("**当前状态**")
            dict_count = len(manager.list_dictionaries())
            active_count = len(manager.active_dictionaries)
            total_words = sum(
                manager.get_dictionary(name).size() 
                for name in manager.list_dictionaries()
                if manager.get_dictionary(name)
            )
            
            st.metric("词典数量", dict_count)
            st.metric("激活词典", active_count)
            st.metric("总词汇数", total_words)


def render_dictionary_manager_compact():
    """
    渲染紧凑版词典管理界面（用于文本预处理页面）
    
    提供简化的词典管理功能：
    - 快速添加词汇
    - 激活/停用词典
    - 查看当前状态
    """
    import streamlit as st
    import pandas as pd
    from utils.session_state import log_message
    
    # 获取词典管理器
    if st.session_state.get("dictionary_manager") is None:
        st.session_state["dictionary_manager"] = DictionaryManager()
    
    manager: DictionaryManager = st.session_state["dictionary_manager"]
    
    # 显示当前状态
    dict_names = manager.list_dictionaries()
    active_dicts = manager.active_dictionaries
    
    col1, col2, col3 = st.columns(3)
    col1.metric("词典数量", len(dict_names))
    col2.metric("激活词典", len(active_dicts))
    col3.metric("总词汇数", sum(
        manager.get_dictionary(name).size() 
        for name in dict_names
        if manager.get_dictionary(name)
    ))
    
    # 快速操作区
    tabs = st.tabs(["快速添加", "词典管理", "导入词典"])
    
    # 快速添加词汇
    with tabs[0]:
        col1, col2 = st.columns([3, 1])
        with col1:
            quick_words = st.text_area(
                "输入专业词汇（每行一个，可选词性用空格分隔）",
                height=100,
                placeholder="人工智能 n\n科技创新\n数字经济",
                key="compact_quick_words"
            )
        with col2:
            # 选择目标词典
            if dict_names:
                target_dict = st.selectbox(
                    "目标词典",
                    dict_names,
                    key="compact_target_dict"
                )
            else:
                target_dict = None
                st.info("请先创建词典")
            
            # 创建新词典
            new_dict_name = st.text_input("或创建新词典", key="compact_new_dict")
            if st.button("创建", key="compact_create_dict"):
                if new_dict_name:
                    result = manager.create_dictionary(new_dict_name)
                    if result:
                        manager.activate_dictionary(new_dict_name)
                        st.success(f"已创建并激活: {new_dict_name}")
                        log_message(f"创建词典: {new_dict_name}")
                        st.rerun()
        
        if st.button("添加词汇", key="compact_add_words", type="primary"):
            if quick_words and (target_dict or new_dict_name):
                dict_name = target_dict if target_dict else new_dict_name
                
                # 如果是新词典，先创建
                if dict_name not in dict_names:
                    manager.create_dictionary(dict_name)
                    manager.activate_dictionary(dict_name)
                
                # 添加词汇
                lines = quick_words.strip().split('\n')
                added = 0
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    word = parts[0]
                    pos = parts[1] if len(parts) > 1 else None
                    if manager.add_word(dict_name, word, pos):
                        added += 1
                
                st.success(f"已添加 {added} 个词汇到 {dict_name}")
                log_message(f"添加 {added} 个词汇到词典 {dict_name}")
            else:
                st.warning("请输入词汇并选择目标词典")
    
    # 词典管理
    with tabs[1]:
        if dict_names:
            for name in dict_names:
                dictionary = manager.get_dictionary(name)
                if dictionary:
                    is_active = manager.is_active(name)
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        status = "✅" if is_active else "⬜"
                        st.write(f"{status} **{name}** ({dictionary.size()} 词)")
                    
                    with col2:
                        if is_active:
                            if st.button("停用", key=f"compact_deact_{name}"):
                                manager.deactivate_dictionary(name)
                                st.rerun()
                        else:
                            if st.button("激活", key=f"compact_act_{name}"):
                                manager.activate_dictionary(name)
                                st.rerun()
                    
                    with col3:
                        # 查看词汇
                        with st.popover("查看"):
                            words = dictionary.to_list()[:20]
                            if words:
                                for w, p in words:
                                    st.write(f"• {w}" + (f" ({p})" if p else ""))
                                if dictionary.size() > 20:
                                    st.caption(f"...还有 {dictionary.size() - 20} 个词汇")
                    
                    with col4:
                        if st.button("删除", key=f"compact_del_{name}", type="secondary"):
                            manager.remove_dictionary(name)
                            st.rerun()
        else:
            st.info("暂无词典，请在「快速添加」中创建")
    
    # 导入词典
    with tabs[2]:
        uploaded = st.file_uploader(
            "上传词典文件（TXT格式，每行一个词）",
            type=["txt"],
            key="compact_upload_dict"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            import_name = st.text_input("词典名称", key="compact_import_name")
        with col2:
            auto_activate = st.checkbox("导入后自动激活", value=True, key="compact_auto_activate")
        
        if st.button("导入", key="compact_import_btn"):
            if uploaded and import_name:
                content = uploaded.read().decode('utf-8', errors='replace')
                if manager.import_dictionary_from_text(content, import_name):
                    if auto_activate:
                        manager.activate_dictionary(import_name)
                    dictionary = manager.get_dictionary(import_name)
                    word_count = dictionary.size() if dictionary else 0
                    st.success(f"已导入 {word_count} 个词汇")
                    log_message(f"导入词典 {import_name}，共 {word_count} 个词汇")
                    st.rerun()
                else:
                    st.error("导入失败")
            else:
                st.warning("请上传文件并输入词典名称")
    
    # 提示信息
    if active_dicts:
        st.success(f"✅ 已激活词典: {', '.join(active_dicts)}（词汇已应用到分词器）")
    else:
        st.warning("⚠️ 未激活任何词典，专业词汇可能无法被正确分词")
