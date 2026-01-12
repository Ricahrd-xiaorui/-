# -*- coding: utf-8 -*-
"""
Property-Based Tests for Comparative Analyzer Module

**Feature: text-processing-enhancement**
**Property 7: 相似度计算对称性**
**Validates: Requirements 5.2**

This module tests the property that:
- For any two documents A and B, similarity(A, B) should equal similarity(B, A)
- For any document A, similarity(A, A) should equal 1.0
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from hypothesis import given, strategies as st, settings, assume
import numpy as np

from modules.comparative_analyzer import ComparativeAnalyzer


# ============================================================================
# Custom Strategies for generating test data
# ============================================================================

# Strategy for generating Chinese words (common policy-related terms)
chinese_word_strategy = st.sampled_from([
    "政策", "发展", "创新", "科技", "经济", "改革", "建设", "实施",
    "推进", "加强", "完善", "提高", "促进", "支持", "保障", "管理",
    "服务", "体系", "机制", "制度", "规划", "目标", "任务", "措施",
    "人才", "资源", "环境", "产业", "企业", "市场", "社会", "国家",
    "数字", "智能", "绿色", "高质量", "现代化", "协调", "开放", "共享"
])

# Strategy for generating tokenized text (list of words)
tokenized_text_strategy = st.lists(
    chinese_word_strategy,
    min_size=5,
    max_size=50
)

# Strategy for generating document names
doc_name_strategy = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_"),
    min_size=3,
    max_size=15
).map(lambda s: f"doc_{s}.txt")


@st.composite
def two_documents_strategy(draw):
    """
    Generate two documents with tokenized texts and names.
    Returns: (texts, file_names)
    """
    text1 = draw(tokenized_text_strategy)
    text2 = draw(tokenized_text_strategy)
    
    # Ensure texts are non-empty
    assume(len(text1) > 0 and len(text2) > 0)
    
    texts = [text1, text2]
    file_names = ["doc_a.txt", "doc_b.txt"]
    
    return texts, file_names


@st.composite
def multiple_documents_strategy(draw):
    """
    Generate multiple documents with tokenized texts and names.
    Returns: (texts, file_names)
    """
    num_docs = draw(st.integers(min_value=2, max_value=8))
    
    texts = []
    file_names = []
    
    for i in range(num_docs):
        text = draw(tokenized_text_strategy)
        assume(len(text) > 0)
        texts.append(text)
        file_names.append(f"doc_{i}.txt")
    
    return texts, file_names


@st.composite
def single_document_strategy(draw):
    """
    Generate a single document for self-similarity testing.
    Returns: (texts, file_names)
    """
    text = draw(tokenized_text_strategy)
    assume(len(text) > 0)
    
    texts = [text]
    file_names = ["doc_single.txt"]
    
    return texts, file_names


# ============================================================================
# Property Tests
# ============================================================================

class TestSimilaritySymmetryProperty:
    """
    Property 7: 相似度计算对称性
    
    *For any* 两个文档A和B，similarity(A, B) 应等于 similarity(B, A)，
    且 similarity(A, A) 应等于 1.0。
    **Validates: Requirements 5.2**
    """

    @given(data=two_documents_strategy())
    @settings(max_examples=100, deadline=None)
    def test_cosine_similarity_symmetry(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 7: 相似度计算对称性**
        **Validates: Requirements 5.2**
        
        Property: For any two documents A and B, cosine similarity(A, B) 
        should equal similarity(B, A).
        """
        texts, file_names = data
        
        analyzer = ComparativeAnalyzer(texts, file_names)
        
        # Calculate similarity in both directions
        sim_ab = analyzer.calculate_similarity(0, 1, method='cosine')
        sim_ba = analyzer.calculate_similarity(1, 0, method='cosine')
        
        # Verify symmetry (allow small floating point tolerance)
        assert abs(sim_ab - sim_ba) < 1e-10, (
            f"Cosine similarity is not symmetric: "
            f"sim(A,B)={sim_ab} != sim(B,A)={sim_ba}"
        )

    @given(data=two_documents_strategy())
    @settings(max_examples=100, deadline=None)
    def test_jaccard_similarity_symmetry(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 7: 相似度计算对称性**
        **Validates: Requirements 5.2**
        
        Property: For any two documents A and B, Jaccard similarity(A, B) 
        should equal similarity(B, A).
        """
        texts, file_names = data
        
        analyzer = ComparativeAnalyzer(texts, file_names)
        
        # Calculate similarity in both directions
        sim_ab = analyzer.calculate_similarity(0, 1, method='jaccard')
        sim_ba = analyzer.calculate_similarity(1, 0, method='jaccard')
        
        # Verify symmetry
        assert abs(sim_ab - sim_ba) < 1e-10, (
            f"Jaccard similarity is not symmetric: "
            f"sim(A,B)={sim_ab} != sim(B,A)={sim_ba}"
        )

    @given(data=single_document_strategy())
    @settings(max_examples=100, deadline=None)
    def test_self_similarity_equals_one_cosine(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 7: 相似度计算对称性**
        **Validates: Requirements 5.2**
        
        Property: For any document A, cosine similarity(A, A) should equal 1.0.
        """
        texts, file_names = data
        
        analyzer = ComparativeAnalyzer(texts, file_names)
        
        # Calculate self-similarity
        self_sim = analyzer.calculate_similarity(0, 0, method='cosine')
        
        # Verify self-similarity is 1.0
        assert abs(self_sim - 1.0) < 1e-10, (
            f"Self-similarity should be 1.0, got {self_sim}"
        )

    @given(data=single_document_strategy())
    @settings(max_examples=100, deadline=None)
    def test_self_similarity_equals_one_jaccard(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 7: 相似度计算对称性**
        **Validates: Requirements 5.2**
        
        Property: For any document A, Jaccard similarity(A, A) should equal 1.0.
        """
        texts, file_names = data
        
        analyzer = ComparativeAnalyzer(texts, file_names)
        
        # Calculate self-similarity
        self_sim = analyzer.calculate_similarity(0, 0, method='jaccard')
        
        # Verify self-similarity is 1.0
        assert abs(self_sim - 1.0) < 1e-10, (
            f"Self-similarity should be 1.0, got {self_sim}"
        )

    @given(data=multiple_documents_strategy())
    @settings(max_examples=100, deadline=None)
    def test_similarity_matrix_symmetry_cosine(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 7: 相似度计算对称性**
        **Validates: Requirements 5.2**
        
        Property: The cosine similarity matrix should be symmetric.
        """
        texts, file_names = data
        
        analyzer = ComparativeAnalyzer(texts, file_names)
        
        # Calculate similarity matrix
        sim_matrix = analyzer.calculate_similarity_matrix(method='cosine')
        
        # Verify matrix is symmetric
        n = len(sim_matrix)
        for i in range(n):
            for j in range(n):
                assert abs(sim_matrix[i][j] - sim_matrix[j][i]) < 1e-10, (
                    f"Similarity matrix is not symmetric at ({i},{j}): "
                    f"{sim_matrix[i][j]} != {sim_matrix[j][i]}"
                )

    @given(data=multiple_documents_strategy())
    @settings(max_examples=100, deadline=None)
    def test_similarity_matrix_symmetry_jaccard(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 7: 相似度计算对称性**
        **Validates: Requirements 5.2**
        
        Property: The Jaccard similarity matrix should be symmetric.
        """
        texts, file_names = data
        
        analyzer = ComparativeAnalyzer(texts, file_names)
        
        # Calculate similarity matrix
        sim_matrix = analyzer.calculate_similarity_matrix(method='jaccard')
        
        # Verify matrix is symmetric
        n = len(sim_matrix)
        for i in range(n):
            for j in range(n):
                assert abs(sim_matrix[i][j] - sim_matrix[j][i]) < 1e-10, (
                    f"Jaccard similarity matrix is not symmetric at ({i},{j}): "
                    f"{sim_matrix[i][j]} != {sim_matrix[j][i]}"
                )

    @given(data=multiple_documents_strategy())
    @settings(max_examples=100, deadline=None)
    def test_similarity_matrix_diagonal_equals_one(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 7: 相似度计算对称性**
        **Validates: Requirements 5.2**
        
        Property: The diagonal of the similarity matrix should all be 1.0.
        """
        texts, file_names = data
        
        analyzer = ComparativeAnalyzer(texts, file_names)
        
        # Calculate similarity matrix
        sim_matrix = analyzer.calculate_similarity_matrix(method='cosine')
        
        # Verify diagonal is all 1.0
        n = len(sim_matrix)
        for i in range(n):
            assert abs(sim_matrix[i][i] - 1.0) < 1e-10, (
                f"Diagonal element [{i}][{i}] should be 1.0, got {sim_matrix[i][i]}"
            )

    @given(data=two_documents_strategy())
    @settings(max_examples=100, deadline=None)
    def test_similarity_in_valid_range(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 7: 相似度计算对称性**
        **Validates: Requirements 5.2**
        
        Property: Similarity values should be in the range [0, 1].
        """
        texts, file_names = data
        
        analyzer = ComparativeAnalyzer(texts, file_names)
        
        # Test both methods
        # Allow small floating-point tolerance (1e-10) for values slightly above 1.0
        tolerance = 1e-10
        for method in ['cosine', 'jaccard']:
            sim = analyzer.calculate_similarity(0, 1, method=method)
            
            assert -tolerance <= sim <= 1.0 + tolerance, (
                f"{method} similarity should be in [0, 1], got {sim}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
