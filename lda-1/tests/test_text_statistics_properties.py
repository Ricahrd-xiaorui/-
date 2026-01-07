# -*- coding: utf-8 -*-
"""
Property-Based Tests for Text Statistics Module

**Feature: text-processing-enhancement**
**Property 13: 文本统计一致性**
**Property 14: TTR范围**
**Validates: Requirements 8.1, 8.2**

This module tests the properties that:
- For any text, char_count >= word_count >= sentence_count >= paragraph_count
- For any text, TTR value should be in range (0, 1]
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from hypothesis import given, strategies as st, settings, assume

from modules.text_statistics import TextStatistics, create_text_statistics


# ============================================================================
# Custom Strategies for generating test data
# ============================================================================

# Strategy for generating Chinese text content
chinese_chars = list("人工智能科技创新政策发展经济数字乡村振兴战略实施推进测试文本内容分析研究方法数据处理结果")
punctuation = list("。！？；")
sentence_separators = list("。！？；")

# Strategy for generating tokenized text (list of words)
word_strategy = st.text(
    alphabet=st.sampled_from(chinese_chars),
    min_size=1,
    max_size=10
).filter(lambda x: x.strip() != "")

# Strategy for generating non-empty tokenized text
non_empty_tokenized_strategy = st.lists(
    word_strategy,
    min_size=1,
    max_size=100
)


@st.composite
def consistent_text_strategy(draw):
    """
    Generate consistent raw_text and tokenized_text pairs.
    The tokenized_text is derived from the raw_text to ensure consistency.
    """
    # Generate a list of words (these will be our tokens)
    words = draw(st.lists(word_strategy, min_size=1, max_size=50))
    
    # Generate sentence structure: how many words per sentence
    num_sentences = draw(st.integers(min_value=1, max_value=max(1, len(words))))
    
    # Distribute words across sentences
    sentences = []
    words_remaining = words.copy()
    
    for i in range(num_sentences):
        if not words_remaining:
            break
        # Take some words for this sentence
        words_in_sentence = draw(st.integers(min_value=1, max_value=max(1, len(words_remaining))))
        sentence_words = words_remaining[:words_in_sentence]
        words_remaining = words_remaining[words_in_sentence:]
        if sentence_words:
            sentences.append("".join(sentence_words))
    
    # Add remaining words to last sentence if any
    if words_remaining and sentences:
        sentences[-1] += "".join(words_remaining)
    elif words_remaining:
        sentences.append("".join(words_remaining))
    
    # Generate paragraph structure
    num_paragraphs = draw(st.integers(min_value=1, max_value=max(1, len(sentences))))
    
    # Distribute sentences across paragraphs
    paragraphs = []
    sentences_remaining = sentences.copy()
    
    for i in range(num_paragraphs):
        if not sentences_remaining:
            break
        sentences_in_para = draw(st.integers(min_value=1, max_value=max(1, len(sentences_remaining))))
        para_sentences = sentences_remaining[:sentences_in_para]
        sentences_remaining = sentences_remaining[sentences_in_para:]
        if para_sentences:
            # Join sentences with sentence-ending punctuation
            para_text = "。".join(para_sentences) + "。"
            paragraphs.append(para_text)
    
    # Add remaining sentences to last paragraph if any
    if sentences_remaining and paragraphs:
        paragraphs[-1] = paragraphs[-1][:-1] + "。".join(sentences_remaining) + "。"
    elif sentences_remaining:
        paragraphs.append("。".join(sentences_remaining) + "。")
    
    # Join paragraphs with double newlines
    raw_text = "\n\n".join(paragraphs)
    
    return raw_text, words


# ============================================================================
# Property Tests
# ============================================================================

class TestTextStatisticsConsistencyProperty:
    """
    Property 13: 文本统计一致性
    
    *For any* 文本，字符数应大于等于词语数，词语数应大于等于句子数，句子数应大于等于段落数。
    **Validates: Requirements 8.1**
    """

    @given(
        text_data=consistent_text_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_char_count_gte_word_count(
        self, text_data: tuple
    ):
        """
        **Feature: text-processing-enhancement, Property 13: 文本统计一致性**
        **Validates: Requirements 8.1**
        
        Property: For any text, character count should be >= word count.
        """
        raw_text, tokenized_text = text_data
        stats = TextStatistics(raw_text=raw_text, tokenized_text=tokenized_text)
        
        char_count = stats.count_characters()
        word_count = stats.count_words()
        
        assert char_count >= word_count, (
            f"Character count ({char_count}) should be >= word count ({word_count})"
        )

    @given(
        text_data=consistent_text_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_word_count_gte_sentence_count(
        self, text_data: tuple
    ):
        """
        **Feature: text-processing-enhancement, Property 13: 文本统计一致性**
        **Validates: Requirements 8.1**
        
        Property: For any text, word count should be >= sentence count.
        """
        raw_text, tokenized_text = text_data
        stats = TextStatistics(raw_text=raw_text, tokenized_text=tokenized_text)
        
        word_count = stats.count_words()
        sentence_count = stats.count_sentences()
        
        assert word_count >= sentence_count, (
            f"Word count ({word_count}) should be >= sentence count ({sentence_count})"
        )

    @given(
        text_data=consistent_text_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_sentence_count_gte_paragraph_count(
        self, text_data: tuple
    ):
        """
        **Feature: text-processing-enhancement, Property 13: 文本统计一致性**
        **Validates: Requirements 8.1**
        
        Property: For any text, sentence count should be >= paragraph count.
        """
        raw_text, tokenized_text = text_data
        stats = TextStatistics(raw_text=raw_text, tokenized_text=tokenized_text)
        
        sentence_count = stats.count_sentences()
        paragraph_count = stats.count_paragraphs()
        
        assert sentence_count >= paragraph_count, (
            f"Sentence count ({sentence_count}) should be >= paragraph count ({paragraph_count})"
        )

    @given(
        text_data=consistent_text_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_full_hierarchy_consistency(
        self, text_data: tuple
    ):
        """
        **Feature: text-processing-enhancement, Property 13: 文本统计一致性**
        **Validates: Requirements 8.1**
        
        Property: For any text, the full hierarchy should hold:
        char_count >= word_count >= sentence_count >= paragraph_count
        """
        raw_text, tokenized_text = text_data
        stats = TextStatistics(raw_text=raw_text, tokenized_text=tokenized_text)
        
        char_count = stats.count_characters()
        word_count = stats.count_words()
        sentence_count = stats.count_sentences()
        paragraph_count = stats.count_paragraphs()
        
        assert char_count >= word_count >= sentence_count >= paragraph_count, (
            f"Hierarchy violated: chars={char_count}, words={word_count}, "
            f"sentences={sentence_count}, paragraphs={paragraph_count}"
        )


class TestTTRRangeProperty:
    """
    Property 14: TTR范围
    
    *For any* 文本的词汇丰富度(TTR)计算结果，TTR值应在(0, 1]范围内。
    **Validates: Requirements 8.2**
    """

    @given(
        tokenized_text=non_empty_tokenized_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_ttr_in_valid_range(
        self, tokenized_text: list
    ):
        """
        **Feature: text-processing-enhancement, Property 14: TTR范围**
        **Validates: Requirements 8.2**
        
        Property: For any non-empty text, TTR should be in range (0, 1].
        """
        assume(len(tokenized_text) > 0)
        
        stats = TextStatistics(raw_text="", tokenized_text=tokenized_text)
        ttr = stats.calculate_ttr()
        
        assert 0 < ttr <= 1, (
            f"TTR ({ttr}) should be in range (0, 1] for non-empty text"
        )

    @given(
        tokenized_text=non_empty_tokenized_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_ttr_upper_bound(
        self, tokenized_text: list
    ):
        """
        **Feature: text-processing-enhancement, Property 14: TTR范围**
        **Validates: Requirements 8.2**
        
        Property: TTR should never exceed 1.0 (unique words / total words <= 1).
        """
        assume(len(tokenized_text) > 0)
        
        stats = TextStatistics(raw_text="", tokenized_text=tokenized_text)
        ttr = stats.calculate_ttr()
        
        assert ttr <= 1.0, (
            f"TTR ({ttr}) should never exceed 1.0"
        )

    @given(
        tokenized_text=non_empty_tokenized_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_ttr_lower_bound_positive(
        self, tokenized_text: list
    ):
        """
        **Feature: text-processing-enhancement, Property 14: TTR范围**
        **Validates: Requirements 8.2**
        
        Property: TTR should be positive for non-empty text.
        """
        assume(len(tokenized_text) > 0)
        
        stats = TextStatistics(raw_text="", tokenized_text=tokenized_text)
        ttr = stats.calculate_ttr()
        
        assert ttr > 0, (
            f"TTR ({ttr}) should be positive for non-empty text"
        )

    @given(
        word=word_strategy,
        repeat_count=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100, deadline=None)
    def test_ttr_with_repeated_words(
        self, word: str, repeat_count: int
    ):
        """
        **Feature: text-processing-enhancement, Property 14: TTR范围**
        **Validates: Requirements 8.2**
        
        Property: For text with only one unique word repeated n times,
        TTR should equal 1/n.
        """
        word = word.strip()
        assume(len(word) >= 1)
        
        tokenized_text = [word] * repeat_count
        stats = TextStatistics(raw_text="", tokenized_text=tokenized_text)
        ttr = stats.calculate_ttr()
        
        expected_ttr = 1.0 / repeat_count
        
        assert abs(ttr - expected_ttr) < 1e-9, (
            f"TTR ({ttr}) should equal 1/{repeat_count} = {expected_ttr} "
            f"for {repeat_count} repetitions of the same word"
        )

    @given(
        words=st.lists(word_strategy, min_size=1, max_size=50, unique=True)
    )
    @settings(max_examples=100, deadline=None)
    def test_ttr_with_all_unique_words(
        self, words: list
    ):
        """
        **Feature: text-processing-enhancement, Property 14: TTR范围**
        **Validates: Requirements 8.2**
        
        Property: For text with all unique words, TTR should equal 1.0.
        """
        # Filter to ensure all words are unique and non-empty
        unique_words = list(set(w.strip() for w in words if w.strip()))
        assume(len(unique_words) >= 1)
        
        stats = TextStatistics(raw_text="", tokenized_text=unique_words)
        ttr = stats.calculate_ttr()
        
        assert abs(ttr - 1.0) < 1e-9, (
            f"TTR ({ttr}) should equal 1.0 for all unique words"
        )

    def test_ttr_empty_text_returns_zero(self):
        """
        **Feature: text-processing-enhancement, Property 14: TTR范围**
        **Validates: Requirements 8.2**
        
        Edge case: For empty text, TTR should return 0.0.
        """
        stats = TextStatistics(raw_text="", tokenized_text=[])
        ttr = stats.calculate_ttr()
        
        assert ttr == 0.0, (
            f"TTR ({ttr}) should be 0.0 for empty text"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
