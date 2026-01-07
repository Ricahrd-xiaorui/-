# -*- coding: utf-8 -*-
"""
Property-Based Tests for Dictionary Manager Module

**Feature: text-processing-enhancement, Property 15: 词典导入导出一致性**
**Validates: Requirements 9.1, 9.9, 9.10**

This module tests the property that for any dictionary, exporting to a file
and then importing should result in a dictionary containing the same words.
"""

import os
import sys
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from hypothesis import given, strategies as st, settings, assume
import jieba

from modules.dictionary_manager import Dictionary, DictionaryManager


# ============================================================================
# Custom Strategies for generating test data
# ============================================================================

# Strategy for generating valid Chinese/English words (non-empty, no whitespace-only)
word_strategy = st.text(
    alphabet=st.sampled_from(
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") +
        list("人工智能科技创新政策发展经济数字乡村振兴战略实施推进")
    ),
    min_size=1,
    max_size=20
).filter(lambda x: x.strip() != "")

# Strategy for generating optional POS tags
pos_strategy = st.one_of(
    st.none(),
    st.sampled_from(["n", "v", "adj", "adv", "nr", "ns", "nt", "nz", "vn"])
)

# Strategy for generating dictionary names (non-empty, no special chars)
dict_name_strategy = st.text(
    alphabet=st.sampled_from(list("abcdefghijklmnopqrstuvwxyz_0123456789")),
    min_size=1,
    max_size=30
).filter(lambda x: x.strip() != "")

# Strategy for generating word-pos pairs
word_pos_pair_strategy = st.tuples(word_strategy, pos_strategy)

# Strategy for generating a list of word-pos pairs (dictionary content)
dict_content_strategy = st.lists(
    word_pos_pair_strategy,
    min_size=0,
    max_size=50
)


# ============================================================================
# Property Tests
# ============================================================================

class TestDictionaryImportExportProperty:
    """
    Property 15: 词典导入导出一致性
    
    *For any* 专业词典，导出到文件后再导入，应得到包含相同词汇的词典。
    **Validates: Requirements 9.1, 9.9, 9.10**
    """

    @given(
        dict_name=dict_name_strategy,
        words_with_pos=dict_content_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_dictionary_export_import_roundtrip_txt(
        self, dict_name: str, words_with_pos: list
    ):
        """
        **Feature: text-processing-enhancement, Property 15: 词典导入导出一致性**
        **Validates: Requirements 9.1, 9.9, 9.10**
        
        Property: For any dictionary, exporting to TXT file and then importing
        should result in a dictionary containing the same words.
        """
        # Create a dictionary manager and add a dictionary
        manager = DictionaryManager()
        dictionary = manager.create_dictionary(dict_name)
        assume(dictionary is not None)
        
        # Add words to the dictionary
        original_words = set()
        for word, pos in words_with_pos:
            word = word.strip()
            if word:  # Only add non-empty words
                dictionary.add(word, pos)
                original_words.add(word)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        ) as tmp:
            tmp_path = tmp.name
        
        try:
            # Export the dictionary
            export_success = manager.export_dictionary(dict_name, tmp_path)
            assert export_success, "Export should succeed"
            
            # Create a new manager and import the dictionary
            new_manager = DictionaryManager()
            import_success = new_manager.import_dictionary(
                tmp_path, f"{dict_name}_imported"
            )
            assert import_success, "Import should succeed"
            
            # Get the imported dictionary
            imported_dict = new_manager.get_dictionary(f"{dict_name}_imported")
            assert imported_dict is not None, "Imported dictionary should exist"
            
            # Verify the words match
            imported_words = imported_dict.get_words()
            
            # The imported dictionary should contain all original words
            assert original_words == imported_words, (
                f"Words mismatch: original={original_words}, imported={imported_words}"
            )
            
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @given(
        dict_name=dict_name_strategy,
        words_with_pos=dict_content_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_dictionary_save_load_roundtrip_json(
        self, dict_name: str, words_with_pos: list
    ):
        """
        **Feature: text-processing-enhancement, Property 15: 词典导入导出一致性**
        **Validates: Requirements 9.10**
        
        Property: For any dictionary manager state, saving to JSON and loading
        should result in equivalent dictionaries.
        """
        # Create a dictionary manager and add a dictionary
        manager = DictionaryManager()
        dictionary = manager.create_dictionary(dict_name)
        assume(dictionary is not None)
        
        # Add words to the dictionary
        original_words = {}
        for word, pos in words_with_pos:
            word = word.strip()
            if word:  # Only add non-empty words
                dictionary.add(word, pos)
                original_words[word] = pos
        
        # Save to temporary JSON file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, encoding='utf-8'
        ) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save all dictionaries
            save_success = manager.save_all(tmp_path)
            assert save_success, "Save should succeed"
            
            # Create a new manager and load
            new_manager = DictionaryManager()
            load_success = new_manager.load_all(tmp_path)
            assert load_success, "Load should succeed"
            
            # Verify the dictionary exists
            loaded_dict = new_manager.get_dictionary(dict_name)
            assert loaded_dict is not None, "Loaded dictionary should exist"
            
            # Verify the words match
            loaded_words = loaded_dict.get_words()
            original_word_set = set(original_words.keys())
            
            assert original_word_set == loaded_words, (
                f"Words mismatch: original={original_word_set}, loaded={loaded_words}"
            )
            
            # Verify POS tags match for each word
            for word, expected_pos in original_words.items():
                actual_pos = loaded_dict.get_pos(word)
                assert actual_pos == expected_pos, (
                    f"POS mismatch for '{word}': expected={expected_pos}, actual={actual_pos}"
                )
            
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestDictionaryTokenizationProperty:
    """
    Property 16: 词典分词效果
    
    *For any* 导入的专业词典词汇，在包含该词汇的文本上进行分词，
    分词结果应包含该词汇作为独立词语。
    **Validates: Requirements 9.3**
    """

    @given(
        dict_name=dict_name_strategy,
        word=word_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_dictionary_word_appears_in_tokenization(
        self, dict_name: str, word: str
    ):
        """
        **Feature: text-processing-enhancement, Property 16: 词典分词效果**
        **Validates: Requirements 9.3**
        
        Property: For any word added to an active dictionary, when tokenizing
        a text containing that word, the tokenization result should include
        that word as an independent token.
        """
        word = word.strip()
        assume(len(word) >= 2)  # Only test words with at least 2 characters
        
        # Create a dictionary manager and add a dictionary with the word
        manager = DictionaryManager()
        dictionary = manager.create_dictionary(dict_name)
        assume(dictionary is not None)
        
        # Add the word to the dictionary
        dictionary.add(word)
        
        # Activate the dictionary (this applies words to jieba)
        manager.activate_dictionary(dict_name)
        
        # Create a text that contains the word
        # Use surrounding context to make it more realistic
        text = f"这是一段包含{word}的测试文本，用于验证分词效果。"
        
        # Tokenize the text using jieba
        tokens = list(jieba.cut(text))
        
        # The word should appear as an independent token
        assert word in tokens, (
            f"Word '{word}' should appear as independent token in tokenization. "
            f"Tokens: {tokens}"
        )

    @given(
        dict_name=dict_name_strategy,
        words=st.lists(word_strategy, min_size=1, max_size=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_multiple_dictionary_words_in_tokenization(
        self, dict_name: str, words: list
    ):
        """
        **Feature: text-processing-enhancement, Property 16: 词典分词效果**
        **Validates: Requirements 9.3**
        
        Property: For any set of words added to an active dictionary, when
        tokenizing a text containing those words, each word should appear
        as an independent token in the result.
        """
        # Filter and deduplicate words
        unique_words = list(set(w.strip() for w in words if len(w.strip()) >= 2))
        assume(len(unique_words) >= 1)
        
        # Create a dictionary manager and add a dictionary with the words
        manager = DictionaryManager()
        dictionary = manager.create_dictionary(dict_name)
        assume(dictionary is not None)
        
        # Add all words to the dictionary
        for word in unique_words:
            dictionary.add(word)
        
        # Activate the dictionary (this applies words to jieba)
        manager.activate_dictionary(dict_name)
        
        # Create a text that contains all the words
        text = "测试文本：" + "，".join(unique_words) + "。这些都是专业术语。"
        
        # Tokenize the text using jieba
        tokens = list(jieba.cut(text))
        
        # Each word should appear as an independent token
        for word in unique_words:
            assert word in tokens, (
                f"Word '{word}' should appear as independent token. "
                f"Tokens: {tokens}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



class TestTermFrequencyProperty:
    """
    Property 17: 术语频率统计
    
    *For any* 专业词典和文本集合，术语频率统计结果中每个术语的频率应等于该术语在所有文本中出现的实际次数。
    **Validates: Requirements 9.8**
    """

    @given(
        dict_name=dict_name_strategy,
        words=st.lists(word_strategy, min_size=1, max_size=10),
        texts=st.lists(
            st.text(
                alphabet=st.sampled_from(
                    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") +
                    list("人工智能科技创新政策发展经济数字乡村振兴战略实施推进测试文本") +
                    list("，。、；：！？")
                ),
                min_size=0,
                max_size=200
            ),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_term_frequency_equals_actual_count(
        self, dict_name: str, words: list, texts: list
    ):
        """
        **Feature: text-processing-enhancement, Property 17: 术语频率统计**
        **Validates: Requirements 9.8**
        
        Property: For any dictionary and text collection, the term frequency
        statistics should equal the actual number of occurrences of each term
        across all texts.
        """
        import re
        
        # Filter and deduplicate words
        unique_words = list(set(w.strip() for w in words if w.strip()))
        assume(len(unique_words) >= 1)
        
        # Create a dictionary manager and add a dictionary with the words
        manager = DictionaryManager()
        dictionary = manager.create_dictionary(dict_name)
        assume(dictionary is not None)
        
        # Add all words to the dictionary
        for word in unique_words:
            dictionary.add(word)
        
        # Activate the dictionary
        manager.activate_dictionary(dict_name)
        
        # Calculate term frequency using the manager
        frequency_result = manager.count_term_frequency(texts)
        
        # Manually calculate expected frequency for each word
        for word in unique_words:
            expected_count = 0
            for text in texts:
                # Count occurrences using the same method as the implementation
                count = len(re.findall(re.escape(word), text))
                expected_count += count
            
            actual_count = frequency_result.get(word, 0)
            
            assert actual_count == expected_count, (
                f"Frequency mismatch for '{word}': "
                f"expected={expected_count}, actual={actual_count}"
            )

    @given(
        dict_name=dict_name_strategy,
        word=word_strategy,
        repeat_count=st.integers(min_value=0, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_term_frequency_with_known_count(
        self, dict_name: str, word: str, repeat_count: int
    ):
        """
        **Feature: text-processing-enhancement, Property 17: 术语频率统计**
        **Validates: Requirements 9.8**
        
        Property: For a word repeated a known number of times in a text,
        the frequency count should exactly match that number.
        """
        word = word.strip()
        assume(len(word) >= 1)
        
        # Create a dictionary manager and add a dictionary with the word
        manager = DictionaryManager()
        dictionary = manager.create_dictionary(dict_name)
        assume(dictionary is not None)
        
        # Add the word to the dictionary
        dictionary.add(word)
        
        # Activate the dictionary
        manager.activate_dictionary(dict_name)
        
        # Create a text with the word repeated exactly repeat_count times
        # Use a separator that won't be part of any word
        separator = "。"
        text = separator.join([word] * repeat_count) if repeat_count > 0 else "空文本"
        
        # Calculate term frequency
        frequency_result = manager.count_term_frequency([text])
        
        actual_count = frequency_result.get(word, 0)
        
        assert actual_count == repeat_count, (
            f"Frequency mismatch for '{word}': "
            f"expected={repeat_count}, actual={actual_count}"
        )

    @given(
        dict_name=dict_name_strategy,
        words=st.lists(word_strategy, min_size=1, max_size=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_term_frequency_sum_across_texts(
        self, dict_name: str, words: list
    ):
        """
        **Feature: text-processing-enhancement, Property 17: 术语频率统计**
        **Validates: Requirements 9.8**
        
        Property: The total frequency of a term across multiple texts should
        equal the sum of its frequencies in each individual text.
        """
        import re
        
        # Filter and deduplicate words
        unique_words = list(set(w.strip() for w in words if w.strip()))
        assume(len(unique_words) >= 1)
        
        # Create a dictionary manager and add a dictionary with the words
        manager = DictionaryManager()
        dictionary = manager.create_dictionary(dict_name)
        assume(dictionary is not None)
        
        # Add all words to the dictionary
        for word in unique_words:
            dictionary.add(word)
        
        # Activate the dictionary
        manager.activate_dictionary(dict_name)
        
        # Create multiple texts, each containing some of the words
        texts = []
        for i, word in enumerate(unique_words):
            # Each text contains the word (i+1) times
            text = f"文本{i}：" + "，".join([word] * (i + 1)) + "。"
            texts.append(text)
        
        # Calculate term frequency using the manager
        frequency_result = manager.count_term_frequency(texts)
        
        # Verify each word's frequency
        for i, word in enumerate(unique_words):
            # The word appears (i+1) times in text i
            # But it might also appear in other texts if words overlap
            expected_count = 0
            for text in texts:
                count = len(re.findall(re.escape(word), text))
                expected_count += count
            
            actual_count = frequency_result.get(word, 0)
            
            assert actual_count == expected_count, (
                f"Frequency mismatch for '{word}': "
                f"expected={expected_count}, actual={actual_count}"
            )

    @given(
        dict_name=dict_name_strategy,
        words=st.lists(word_strategy, min_size=1, max_size=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_term_frequency_empty_texts(
        self, dict_name: str, words: list
    ):
        """
        **Feature: text-processing-enhancement, Property 17: 术语频率统计**
        **Validates: Requirements 9.8**
        
        Property: For empty texts, all term frequencies should be zero.
        """
        # Filter and deduplicate words
        unique_words = list(set(w.strip() for w in words if w.strip()))
        assume(len(unique_words) >= 1)
        
        # Create a dictionary manager and add a dictionary with the words
        manager = DictionaryManager()
        dictionary = manager.create_dictionary(dict_name)
        assume(dictionary is not None)
        
        # Add all words to the dictionary
        for word in unique_words:
            dictionary.add(word)
        
        # Activate the dictionary
        manager.activate_dictionary(dict_name)
        
        # Calculate term frequency for empty texts
        frequency_result = manager.count_term_frequency(["", "", ""])
        
        # All frequencies should be zero (or not present in result)
        for word in unique_words:
            actual_count = frequency_result.get(word, 0)
            assert actual_count == 0, (
                f"Frequency for '{word}' should be 0 for empty texts, "
                f"but got {actual_count}"
            )
