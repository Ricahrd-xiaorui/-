# -*- coding: utf-8 -*-
"""
Property-Based Tests for Temporal Analyzer Module

**Feature: text-processing-enhancement**
**Property 6: 时序排序正确性**
**Validates: Requirements 4.2**

This module tests the property that:
- For any set of documents with time labels, sorting by time should produce
  documents in ascending order of their time labels.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from hypothesis import given, strategies as st, settings, assume

from modules.temporal_analyzer import TemporalAnalyzer, TimeLabel


# ============================================================================
# Custom Strategies for generating test data
# ============================================================================

# Strategy for generating year strings
year_strategy = st.integers(min_value=1990, max_value=2030).map(str)

# Strategy for generating year-month strings (YYYY-MM format)
year_month_strategy = st.tuples(
    st.integers(min_value=1990, max_value=2030),
    st.integers(min_value=1, max_value=12)
).map(lambda t: f"{t[0]}-{t[1]:02d}")

# Strategy for generating full date strings (YYYY-MM-DD format)
full_date_strategy = st.tuples(
    st.integers(min_value=1990, max_value=2030),
    st.integers(min_value=1, max_value=12),
    st.integers(min_value=1, max_value=28)  # Use 28 to avoid invalid dates
).map(lambda t: f"{t[0]}-{t[1]:02d}-{t[2]:02d}")

# Strategy for generating Chinese year format (e.g., "2020年")
chinese_year_strategy = st.integers(min_value=1990, max_value=2030).map(lambda y: f"{y}年")

# Strategy for generating Chinese year-month format (e.g., "2020年1月")
chinese_year_month_strategy = st.tuples(
    st.integers(min_value=1990, max_value=2030),
    st.integers(min_value=1, max_value=12)
).map(lambda t: f"{t[0]}年{t[1]}月")

# Combined time label strategy
time_label_strategy = st.one_of(
    year_strategy,
    year_month_strategy,
    full_date_strategy,
    chinese_year_strategy,
    chinese_year_month_strategy
)

# Strategy for generating document names
doc_name_strategy = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_"),
    min_size=3,
    max_size=20
).map(lambda s: f"doc_{s}.txt")

# Strategy for generating tokenized text (list of words)
word_strategy = st.text(
    alphabet=st.sampled_from("人工智能科技创新政策发展经济数字"),
    min_size=1,
    max_size=5
).filter(lambda x: x.strip() != "")

tokenized_text_strategy = st.lists(word_strategy, min_size=1, max_size=20)


@st.composite
def documents_with_time_labels_strategy(draw):
    """
    Generate a set of documents with unique names and time labels.
    Returns: (texts, file_names, time_labels_dict)
    """
    # Generate number of documents
    num_docs = draw(st.integers(min_value=2, max_value=10))
    
    # Generate unique document names
    file_names = []
    for i in range(num_docs):
        name = f"doc_{i}_{draw(st.integers(min_value=1000, max_value=9999))}.txt"
        file_names.append(name)
    
    # Generate tokenized texts for each document
    texts = [draw(tokenized_text_strategy) for _ in range(num_docs)]
    
    # Generate time labels for each document
    time_labels = {}
    for name in file_names:
        time_labels[name] = draw(time_label_strategy)
    
    return texts, file_names, time_labels


@st.composite
def documents_with_consistent_year_labels_strategy(draw):
    """
    Generate documents with year-only time labels for simpler testing.
    Returns: (texts, file_names, time_labels_dict)
    """
    num_docs = draw(st.integers(min_value=2, max_value=10))
    
    file_names = [f"doc_{i}.txt" for i in range(num_docs)]
    texts = [draw(tokenized_text_strategy) for _ in range(num_docs)]
    
    # Generate year labels
    time_labels = {}
    for name in file_names:
        year = draw(st.integers(min_value=2000, max_value=2025))
        time_labels[name] = str(year)
    
    return texts, file_names, time_labels


# ============================================================================
# Property Tests
# ============================================================================

class TestTemporalSortingProperty:
    """
    Property 6: 时序排序正确性
    
    *For any* 带有时间标签的文档集合，按时间排序后文档应按时间标签的升序排列。
    **Validates: Requirements 4.2**
    """

    @given(data=documents_with_consistent_year_labels_strategy())
    @settings(max_examples=100, deadline=None)
    def test_documents_sorted_by_year_ascending(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 6: 时序排序正确性**
        **Validates: Requirements 4.2**
        
        Property: For any set of documents with year labels, sorting by time
        should produce documents in ascending order of years.
        """
        texts, file_names, time_labels = data
        
        # Create analyzer and set time labels
        analyzer = TemporalAnalyzer(texts, file_names)
        analyzer.set_time_labels_batch(time_labels)
        
        # Get sorted documents
        sorted_docs = analyzer.get_documents_sorted_by_time()
        
        # Verify ascending order
        for i in range(len(sorted_docs) - 1):
            current_label = sorted_docs[i][1]
            next_label = sorted_docs[i + 1][1]
            
            # Normalize labels for comparison
            current_sort_key = TimeLabel._normalize_time_label(current_label)
            next_sort_key = TimeLabel._normalize_time_label(next_label)
            
            assert current_sort_key <= next_sort_key, (
                f"Documents not in ascending order: "
                f"'{current_label}' (key: {current_sort_key}) should come before "
                f"'{next_label}' (key: {next_sort_key})"
            )

    @given(data=documents_with_time_labels_strategy())
    @settings(max_examples=100, deadline=None)
    def test_documents_sorted_by_time_ascending_mixed_formats(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 6: 时序排序正确性**
        **Validates: Requirements 4.2**
        
        Property: For any set of documents with mixed time label formats,
        sorting by time should produce documents in ascending order.
        """
        texts, file_names, time_labels = data
        
        analyzer = TemporalAnalyzer(texts, file_names)
        analyzer.set_time_labels_batch(time_labels)
        
        sorted_docs = analyzer.get_documents_sorted_by_time()
        
        # Verify ascending order using normalized sort keys
        for i in range(len(sorted_docs) - 1):
            current_label = sorted_docs[i][1]
            next_label = sorted_docs[i + 1][1]
            
            current_sort_key = TimeLabel._normalize_time_label(current_label)
            next_sort_key = TimeLabel._normalize_time_label(next_label)
            
            assert current_sort_key <= next_sort_key, (
                f"Documents not in ascending order: "
                f"'{current_label}' (key: {current_sort_key}) should come before "
                f"'{next_label}' (key: {next_sort_key})"
            )

    @given(data=documents_with_time_labels_strategy())
    @settings(max_examples=100, deadline=None)
    def test_sorted_documents_contain_all_labeled_docs(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 6: 时序排序正确性**
        **Validates: Requirements 4.2**
        
        Property: The sorted document list should contain all documents
        that have time labels assigned.
        """
        texts, file_names, time_labels = data
        
        analyzer = TemporalAnalyzer(texts, file_names)
        analyzer.set_time_labels_batch(time_labels)
        
        sorted_docs = analyzer.get_documents_sorted_by_time()
        sorted_doc_names = [doc[0] for doc in sorted_docs]
        
        # All labeled documents should be in the sorted list
        for doc_name in time_labels.keys():
            assert doc_name in sorted_doc_names, (
                f"Document '{doc_name}' with time label should be in sorted list"
            )
        
        # Sorted list should only contain labeled documents
        assert len(sorted_docs) == len(time_labels), (
            f"Sorted list length ({len(sorted_docs)}) should equal "
            f"number of labeled documents ({len(time_labels)})"
        )

    @given(data=documents_with_time_labels_strategy())
    @settings(max_examples=100, deadline=None)
    def test_sorted_periods_are_unique_and_ordered(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 6: 时序排序正确性**
        **Validates: Requirements 4.2**
        
        Property: get_sorted_periods should return unique time labels in ascending order.
        """
        texts, file_names, time_labels = data
        
        analyzer = TemporalAnalyzer(texts, file_names)
        analyzer.set_time_labels_batch(time_labels)
        
        sorted_periods = analyzer.get_sorted_periods()
        
        # Check uniqueness
        assert len(sorted_periods) == len(set(sorted_periods)), (
            "Sorted periods should contain unique values"
        )
        
        # Check ascending order
        for i in range(len(sorted_periods) - 1):
            current_key = TimeLabel._normalize_time_label(sorted_periods[i])
            next_key = TimeLabel._normalize_time_label(sorted_periods[i + 1])
            
            assert current_key <= next_key, (
                f"Periods not in ascending order: "
                f"'{sorted_periods[i]}' should come before '{sorted_periods[i + 1]}'"
            )

    @given(
        year1=st.integers(min_value=2000, max_value=2020),
        year2=st.integers(min_value=2000, max_value=2020)
    )
    @settings(max_examples=100, deadline=None)
    def test_time_label_normalization_preserves_order(self, year1: int, year2: int):
        """
        **Feature: text-processing-enhancement, Property 6: 时序排序正确性**
        **Validates: Requirements 4.2**
        
        Property: Time label normalization should preserve chronological order.
        """
        label1 = str(year1)
        label2 = str(year2)
        
        key1 = TimeLabel._normalize_time_label(label1)
        key2 = TimeLabel._normalize_time_label(label2)
        
        if year1 < year2:
            assert key1 < key2, (
                f"Normalized key for {year1} should be less than key for {year2}"
            )
        elif year1 > year2:
            assert key1 > key2, (
                f"Normalized key for {year1} should be greater than key for {year2}"
            )
        else:
            assert key1 == key2, (
                f"Normalized keys for same year should be equal"
            )

    @given(
        year=st.integers(min_value=2000, max_value=2025),
        month1=st.integers(min_value=1, max_value=12),
        month2=st.integers(min_value=1, max_value=12)
    )
    @settings(max_examples=100, deadline=None)
    def test_year_month_sorting_within_same_year(self, year: int, month1: int, month2: int):
        """
        **Feature: text-processing-enhancement, Property 6: 时序排序正确性**
        **Validates: Requirements 4.2**
        
        Property: Within the same year, months should be sorted correctly.
        """
        label1 = f"{year}-{month1:02d}"
        label2 = f"{year}-{month2:02d}"
        
        key1 = TimeLabel._normalize_time_label(label1)
        key2 = TimeLabel._normalize_time_label(label2)
        
        if month1 < month2:
            assert key1 < key2, (
                f"Month {month1} should sort before month {month2} in year {year}"
            )
        elif month1 > month2:
            assert key1 > key2, (
                f"Month {month1} should sort after month {month2} in year {year}"
            )
        else:
            assert key1 == key2, (
                f"Same months should have equal sort keys"
            )


class TestTimeLabelNormalization:
    """
    Additional tests for time label normalization to ensure sorting correctness.
    """

    def test_chinese_year_format_normalization(self):
        """Test that Chinese year format normalizes correctly."""
        label = "2020年"
        key = TimeLabel._normalize_time_label(label)
        assert key == "2020-00-00", f"Expected '2020-00-00', got '{key}'"

    def test_chinese_year_month_format_normalization(self):
        """Test that Chinese year-month format normalizes correctly."""
        label = "2020年5月"
        key = TimeLabel._normalize_time_label(label)
        assert key == "2020-05-00", f"Expected '2020-05-00', got '{key}'"

    def test_standard_date_format_normalization(self):
        """Test that standard date format normalizes correctly."""
        label = "2020-03-15"
        key = TimeLabel._normalize_time_label(label)
        assert key == "2020-03-15", f"Expected '2020-03-15', got '{key}'"

    def test_empty_label_normalization(self):
        """Test that empty label normalizes to high value for sorting last."""
        label = ""
        key = TimeLabel._normalize_time_label(label)
        assert key == "9999", f"Expected '9999', got '{key}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
