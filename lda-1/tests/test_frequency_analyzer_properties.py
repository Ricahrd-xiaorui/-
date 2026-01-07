# -*- coding: utf-8 -*-
"""
Property-Based Tests for Frequency Analyzer Module

**Feature: text-processing-enhancement**
**Property 2: 词频统计正确性**
**Property 3: 共现频率阈值过滤**
**Validates: Requirements 2.1, 2.2, 2.5**

This module tests the properties that:
- For any tokenized text collection, the sum of all word frequencies should equal
  the total word count in the texts
- For any POS-filtered result, each word should belong to the specified POS tags
- For any co-occurrence result and minimum frequency threshold, all word pairs
  in the filtered result should have frequency >= threshold
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from hypothesis import given, strategies as st, settings, assume
from collections import Counter

from modules.frequency_analyzer import FrequencyAnalyzer, CooccurrenceAnalyzer


# ============================================================================
# Custom Strategies for generating test data
# ============================================================================

# Strategy for generating Chinese/English words
chinese_chars = list("人工智能科技创新政策发展经济数字乡村振兴战略实施推进测试文本")
english_chars = list("abcdefghijklmnopqrstuvwxyz")

word_strategy = st.text(
    alphabet=st.sampled_from(chinese_chars + english_chars),
    min_size=1,
    max_size=10
).filter(lambda x: x.strip() != "")

# Strategy for generating POS tags
pos_tags_list = ["n", "v", "a", "d", "nr", "ns", "nt", "nz", "vn", "vd", "ad", "an"]
pos_strategy = st.sampled_from(pos_tags_list)

# Strategy for generating a single tokenized text (list of words)
tokenized_text_strategy = st.lists(
    word_strategy,
    min_size=0,
    max_size=100
)

# Strategy for generating multiple tokenized texts
texts_strategy = st.lists(
    tokenized_text_strategy,
    min_size=1,
    max_size=10
)

# Strategy for generating non-empty texts (at least one word total)
non_empty_texts_strategy = st.lists(
    st.lists(word_strategy, min_size=1, max_size=50),
    min_size=1,
    max_size=10
)


@st.composite
def texts_with_pos_strategy(draw):
    """
    Generate consistent texts and pos_tags pairs.
    Each word in texts has a corresponding POS tag.
    """
    # Generate texts
    num_texts = draw(st.integers(min_value=1, max_value=5))
    texts = []
    pos_tags = []
    
    for _ in range(num_texts):
        num_words = draw(st.integers(min_value=0, max_value=30))
        text_words = []
        text_pos = []
        
        for _ in range(num_words):
            word = draw(word_strategy)
            pos = draw(pos_strategy)
            text_words.append(word)
            text_pos.append(pos)
        
        texts.append(text_words)
        pos_tags.append(text_pos)
    
    return texts, pos_tags


# ============================================================================
# Property Tests
# ============================================================================

class TestWordFrequencySumProperty:
    """
    Property 2: 词频统计正确性 - 词频之和等于总词数
    
    *For any* 分词后的文本集合，词频统计结果中所有词频之和应等于文本中的总词数。
    **Validates: Requirements 2.1**
    """

    @given(texts=texts_strategy)
    @settings(max_examples=100, deadline=None)
    def test_frequency_sum_equals_total_words(self, texts: list):
        """
        **Feature: text-processing-enhancement, Property 2: 词频统计正确性**
        **Validates: Requirements 2.1**
        
        Property: For any tokenized text collection, the sum of all word
        frequencies should equal the total word count in the texts.
        """
        analyzer = FrequencyAnalyzer(texts)
        word_freq = analyzer.calculate_word_frequency()
        
        # Calculate expected total word count
        expected_total = sum(len(text) for text in texts)
        
        # Calculate actual sum of frequencies
        actual_sum = sum(word_freq.values())
        
        assert actual_sum == expected_total, (
            f"Sum of frequencies ({actual_sum}) should equal "
            f"total word count ({expected_total})"
        )

    @given(texts=non_empty_texts_strategy)
    @settings(max_examples=100, deadline=None)
    def test_frequency_sum_with_non_empty_texts(self, texts: list):
        """
        **Feature: text-processing-enhancement, Property 2: 词频统计正确性**
        **Validates: Requirements 2.1**
        
        Property: For non-empty text collections, the frequency sum should
        be positive and equal to total word count.
        """
        analyzer = FrequencyAnalyzer(texts)
        word_freq = analyzer.calculate_word_frequency()
        
        expected_total = sum(len(text) for text in texts)
        actual_sum = sum(word_freq.values())
        
        assert actual_sum > 0, "Frequency sum should be positive for non-empty texts"
        assert actual_sum == expected_total, (
            f"Sum of frequencies ({actual_sum}) should equal "
            f"total word count ({expected_total})"
        )

    @given(texts=texts_strategy)
    @settings(max_examples=100, deadline=None)
    def test_get_total_word_count_consistency(self, texts: list):
        """
        **Feature: text-processing-enhancement, Property 2: 词频统计正确性**
        **Validates: Requirements 2.1**
        
        Property: The get_total_word_count method should return the same
        value as the sum of all word frequencies.
        """
        analyzer = FrequencyAnalyzer(texts)
        word_freq = analyzer.calculate_word_frequency()
        
        total_from_method = analyzer.get_total_word_count()
        total_from_freq_sum = sum(word_freq.values())
        
        assert total_from_method == total_from_freq_sum, (
            f"get_total_word_count ({total_from_method}) should equal "
            f"sum of frequencies ({total_from_freq_sum})"
        )

    @given(
        word=word_strategy,
        repeat_count=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_single_word_frequency(self, word: str, repeat_count: int):
        """
        **Feature: text-processing-enhancement, Property 2: 词频统计正确性**
        **Validates: Requirements 2.1**
        
        Property: For a text with a single word repeated n times,
        that word's frequency should be exactly n.
        """
        word = word.strip()
        assume(len(word) >= 1)
        
        texts = [[word] * repeat_count]
        analyzer = FrequencyAnalyzer(texts)
        word_freq = analyzer.calculate_word_frequency()
        
        assert word in word_freq, f"Word '{word}' should be in frequency dict"
        assert word_freq[word] == repeat_count, (
            f"Frequency of '{word}' ({word_freq[word]}) should be {repeat_count}"
        )


class TestPOSFilterProperty:
    """
    Property 2: 词频统计正确性 - 词性筛选正确性
    
    *For any* 按词性筛选后的结果中每个词都属于指定词性。
    **Validates: Requirements 2.2**
    """

    @given(data=texts_with_pos_strategy())
    @settings(max_examples=100, deadline=None)
    def test_pos_filter_returns_only_specified_pos(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 2: 词频统计正确性**
        **Validates: Requirements 2.2**
        
        Property: For any POS-filtered result, each word should belong
        to the specified POS tags.
        """
        texts, pos_tags = data
        
        # Skip if no words
        total_words = sum(len(t) for t in texts)
        assume(total_words > 0)
        
        analyzer = FrequencyAnalyzer(texts, pos_tags)
        
        # Test filtering by noun POS tags
        noun_pos = ["n", "nr", "ns", "nt", "nz"]
        filtered = analyzer.filter_by_pos(noun_pos)
        
        # Build word-to-pos mapping for verification
        word_pos_map = {}
        for text_idx, text in enumerate(texts):
            if text_idx < len(pos_tags):
                for word_idx, word in enumerate(text):
                    if word_idx < len(pos_tags[text_idx]):
                        pos = pos_tags[text_idx][word_idx]
                        if word not in word_pos_map:
                            word_pos_map[word] = set()
                        word_pos_map[word].add(pos)
        
        # Verify each filtered word has at least one matching POS
        noun_pos_set = set(noun_pos)
        for word in filtered.keys():
            word_pos = word_pos_map.get(word, set())
            has_matching_pos = bool(word_pos & noun_pos_set)
            assert has_matching_pos, (
                f"Word '{word}' with POS {word_pos} should have at least one "
                f"POS in {noun_pos}"
            )

    @given(data=texts_with_pos_strategy())
    @settings(max_examples=100, deadline=None)
    def test_pos_filter_with_verb_tags(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 2: 词频统计正确性**
        **Validates: Requirements 2.2**
        
        Property: Filtering by verb POS tags should only return words
        that have verb POS tags.
        """
        texts, pos_tags = data
        
        total_words = sum(len(t) for t in texts)
        assume(total_words > 0)
        
        analyzer = FrequencyAnalyzer(texts, pos_tags)
        
        # Test filtering by verb POS tags
        verb_pos = ["v", "vd", "vn"]
        filtered = analyzer.filter_by_pos(verb_pos)
        
        # Build word-to-pos mapping for verification
        word_pos_map = {}
        for text_idx, text in enumerate(texts):
            if text_idx < len(pos_tags):
                for word_idx, word in enumerate(text):
                    if word_idx < len(pos_tags[text_idx]):
                        pos = pos_tags[text_idx][word_idx]
                        if word not in word_pos_map:
                            word_pos_map[word] = set()
                        word_pos_map[word].add(pos)
        
        # Verify each filtered word has at least one matching POS
        verb_pos_set = set(verb_pos)
        for word in filtered.keys():
            word_pos = word_pos_map.get(word, set())
            has_matching_pos = bool(word_pos & verb_pos_set)
            assert has_matching_pos, (
                f"Word '{word}' with POS {word_pos} should have at least one "
                f"POS in {verb_pos}"
            )

    @given(data=texts_with_pos_strategy())
    @settings(max_examples=100, deadline=None)
    def test_pos_filter_subset_of_total(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 2: 词频统计正确性**
        **Validates: Requirements 2.2**
        
        Property: The sum of frequencies in POS-filtered results should be
        less than or equal to the total word count.
        """
        texts, pos_tags = data
        
        analyzer = FrequencyAnalyzer(texts, pos_tags)
        
        # Get total frequency
        total_freq = analyzer.calculate_word_frequency()
        total_sum = sum(total_freq.values())
        
        # Get filtered frequency for any POS
        filtered = analyzer.filter_by_pos(["n", "v"])
        filtered_sum = sum(filtered.values())
        
        assert filtered_sum <= total_sum, (
            f"Filtered frequency sum ({filtered_sum}) should be <= "
            f"total frequency sum ({total_sum})"
        )

    def test_pos_filter_without_pos_tags_returns_empty(self):
        """
        **Feature: text-processing-enhancement, Property 2: 词频统计正确性**
        **Validates: Requirements 2.2**
        
        Edge case: When no POS tags are provided, filter_by_pos should
        return an empty dictionary.
        """
        texts = [["word1", "word2", "word3"]]
        analyzer = FrequencyAnalyzer(texts, pos_tags=None)
        
        filtered = analyzer.filter_by_pos(["n", "v"])
        
        assert filtered == {}, (
            "filter_by_pos should return empty dict when no POS tags provided"
        )


class TestFrequencyAnalyzerEdgeCases:
    """
    Additional edge case tests for FrequencyAnalyzer.
    """

    def test_empty_texts_returns_empty_frequency(self):
        """
        Edge case: Empty texts should return empty frequency dictionary.
        """
        analyzer = FrequencyAnalyzer([])
        word_freq = analyzer.calculate_word_frequency()
        
        assert word_freq == {}, "Empty texts should return empty frequency dict"
        assert analyzer.get_total_word_count() == 0, "Total word count should be 0"

    def test_texts_with_empty_lists(self):
        """
        Edge case: Texts containing empty lists should be handled correctly.
        """
        texts = [[], ["word"], [], ["another", "word"], []]
        analyzer = FrequencyAnalyzer(texts)
        word_freq = analyzer.calculate_word_frequency()
        
        expected_total = 3  # "word", "another", "word"
        actual_sum = sum(word_freq.values())
        
        assert actual_sum == expected_total, (
            f"Sum ({actual_sum}) should equal expected ({expected_total})"
        )

    @given(texts=texts_strategy)
    @settings(max_examples=100, deadline=None)
    def test_frequency_values_are_positive(self, texts: list):
        """
        Property: All frequency values should be positive integers.
        """
        analyzer = FrequencyAnalyzer(texts)
        word_freq = analyzer.calculate_word_frequency()
        
        for word, freq in word_freq.items():
            assert isinstance(freq, int), f"Frequency of '{word}' should be int"
            assert freq > 0, f"Frequency of '{word}' should be positive"


# ============================================================================
# Property 3: 共现频率阈值过滤
# ============================================================================

class TestCooccurrenceThresholdFilterProperty:
    """
    Property 3: 共现频率阈值过滤
    
    *For any* 共现分析结果和最小频率阈值，过滤后的结果中每对词语的共现频率都应大于等于阈值。
    **Validates: Requirements 2.5**
    """

    @given(
        texts=non_empty_texts_strategy,
        window_size=st.integers(min_value=2, max_value=10),
        min_freq=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_filtered_cooccurrence_meets_threshold(
        self, texts: list, window_size: int, min_freq: int
    ):
        """
        **Feature: text-processing-enhancement, Property 3: 共现频率阈值过滤**
        **Validates: Requirements 2.5**
        
        Property: For any co-occurrence result and minimum frequency threshold,
        all word pairs in the filtered result should have frequency >= threshold.
        """
        analyzer = CooccurrenceAnalyzer(texts, window_size)
        filtered = analyzer.filter_by_threshold(min_freq)
        
        # Verify all filtered pairs meet the threshold
        for pair, freq in filtered.items():
            assert freq >= min_freq, (
                f"Co-occurrence pair {pair} has frequency {freq} which is "
                f"less than threshold {min_freq}"
            )

    @given(
        texts=non_empty_texts_strategy,
        window_size=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_threshold_one_returns_all_cooccurrences(
        self, texts: list, window_size: int
    ):
        """
        **Feature: text-processing-enhancement, Property 3: 共现频率阈值过滤**
        **Validates: Requirements 2.5**
        
        Property: Filtering with threshold=1 should return all co-occurrences
        (since all co-occurrences have frequency >= 1).
        """
        analyzer = CooccurrenceAnalyzer(texts, window_size)
        all_cooccurrences = analyzer.calculate_cooccurrence()
        filtered = analyzer.filter_by_threshold(1)
        
        assert filtered == all_cooccurrences, (
            "Filtering with threshold=1 should return all co-occurrences"
        )

    @given(
        texts=non_empty_texts_strategy,
        window_size=st.integers(min_value=2, max_value=10),
        threshold1=st.integers(min_value=1, max_value=10),
        threshold2=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_higher_threshold_returns_subset(
        self, texts: list, window_size: int, threshold1: int, threshold2: int
    ):
        """
        **Feature: text-processing-enhancement, Property 3: 共现频率阈值过滤**
        **Validates: Requirements 2.5**
        
        Property: A higher threshold should return a subset of results
        from a lower threshold.
        """
        analyzer = CooccurrenceAnalyzer(texts, window_size)
        
        lower_threshold = min(threshold1, threshold2)
        higher_threshold = max(threshold1, threshold2)
        
        lower_filtered = analyzer.filter_by_threshold(lower_threshold)
        higher_filtered = analyzer.filter_by_threshold(higher_threshold)
        
        # All pairs in higher_filtered should be in lower_filtered
        for pair in higher_filtered.keys():
            assert pair in lower_filtered, (
                f"Pair {pair} in higher threshold result should also be "
                f"in lower threshold result"
            )
        
        # The count of higher_filtered should be <= lower_filtered
        assert len(higher_filtered) <= len(lower_filtered), (
            f"Higher threshold ({higher_threshold}) result count "
            f"({len(higher_filtered)}) should be <= lower threshold "
            f"({lower_threshold}) result count ({len(lower_filtered)})"
        )

    @given(
        texts=non_empty_texts_strategy,
        window_size=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_very_high_threshold_returns_empty_or_subset(
        self, texts: list, window_size: int
    ):
        """
        **Feature: text-processing-enhancement, Property 3: 共现频率阈值过滤**
        **Validates: Requirements 2.5**
        
        Property: A very high threshold should return empty or a small subset.
        """
        analyzer = CooccurrenceAnalyzer(texts, window_size)
        all_cooccurrences = analyzer.calculate_cooccurrence()
        
        # Use a threshold higher than any possible frequency
        very_high_threshold = 10000
        filtered = analyzer.filter_by_threshold(very_high_threshold)
        
        # Should be empty or all remaining pairs have freq >= threshold
        for pair, freq in filtered.items():
            assert freq >= very_high_threshold, (
                f"Pair {pair} has frequency {freq} < threshold {very_high_threshold}"
            )

    @given(
        texts=non_empty_texts_strategy,
        window_size=st.integers(min_value=2, max_value=10),
        min_freq=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_filtered_frequencies_match_original(
        self, texts: list, window_size: int, min_freq: int
    ):
        """
        **Feature: text-processing-enhancement, Property 3: 共现频率阈值过滤**
        **Validates: Requirements 2.5**
        
        Property: The frequency values in filtered results should match
        the original co-occurrence frequencies.
        """
        analyzer = CooccurrenceAnalyzer(texts, window_size)
        all_cooccurrences = analyzer.calculate_cooccurrence()
        filtered = analyzer.filter_by_threshold(min_freq)
        
        # Verify frequencies match
        for pair, freq in filtered.items():
            assert pair in all_cooccurrences, (
                f"Filtered pair {pair} should exist in original co-occurrences"
            )
            assert freq == all_cooccurrences[pair], (
                f"Filtered frequency {freq} for {pair} should match "
                f"original frequency {all_cooccurrences[pair]}"
            )


class TestCooccurrenceAnalyzerEdgeCases:
    """
    Additional edge case tests for CooccurrenceAnalyzer.
    """

    def test_empty_texts_returns_empty_cooccurrence(self):
        """
        Edge case: Empty texts should return empty co-occurrence dictionary.
        """
        analyzer = CooccurrenceAnalyzer([])
        cooccurrence = analyzer.calculate_cooccurrence()
        
        assert cooccurrence == {}, "Empty texts should return empty co-occurrence dict"

    def test_single_word_text_returns_empty_cooccurrence(self):
        """
        Edge case: A text with only one word should have no co-occurrences.
        """
        texts = [["word"]]
        analyzer = CooccurrenceAnalyzer(texts, window_size=5)
        cooccurrence = analyzer.calculate_cooccurrence()
        
        assert cooccurrence == {}, "Single word text should have no co-occurrences"

    def test_same_word_repeated_no_self_cooccurrence(self):
        """
        Edge case: Same word repeated should not create self-co-occurrence.
        """
        texts = [["word", "word", "word"]]
        analyzer = CooccurrenceAnalyzer(texts, window_size=5)
        cooccurrence = analyzer.calculate_cooccurrence()
        
        # Should be empty since same word doesn't co-occur with itself
        assert cooccurrence == {}, (
            "Same word repeated should not create co-occurrence pairs"
        )

    def test_two_different_words_create_cooccurrence(self):
        """
        Edge case: Two different words within window should create co-occurrence.
        """
        texts = [["word1", "word2"]]
        analyzer = CooccurrenceAnalyzer(texts, window_size=5)
        cooccurrence = analyzer.calculate_cooccurrence()
        
        expected_pair = tuple(sorted(["word1", "word2"]))
        assert expected_pair in cooccurrence, (
            f"Expected pair {expected_pair} should be in co-occurrences"
        )
        assert cooccurrence[expected_pair] == 1, (
            f"Co-occurrence frequency should be 1"
        )

    @given(
        window_size=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=50, deadline=None)
    def test_window_size_affects_cooccurrence(self, window_size: int):
        """
        Property: Window size should affect which words are considered co-occurring.
        """
        # Create a text where words are spaced apart
        texts = [["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]]
        analyzer = CooccurrenceAnalyzer(texts, window_size)
        cooccurrence = analyzer.calculate_cooccurrence()
        
        # Check that pairs beyond window size don't co-occur
        for (word1, word2), freq in cooccurrence.items():
            # Find positions
            pos1 = texts[0].index(word1)
            pos2 = texts[0].index(word2)
            distance = abs(pos2 - pos1)
            
            assert distance <= window_size, (
                f"Words {word1} and {word2} at distance {distance} "
                f"should not co-occur with window size {window_size}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
