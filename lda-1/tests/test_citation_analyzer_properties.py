# -*- coding: utf-8 -*-
"""
Property-Based Tests for Citation Analyzer Module

**Feature: text-processing-enhancement**
**Property 9: 引用网络一致性**
**Validates: Requirements 6.3, 6.5**

This module tests the property that:
- For any citation network, a document's cited_by_count should equal the number of 
  incoming edges (edges pointing to that document) in the network
- For any citation network, a document's cites_count should equal the number of 
  outgoing edges (edges from that document) in the network
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from hypothesis import given, strategies as st, settings, assume
from collections import Counter

from modules.citation_analyzer import CitationAnalyzer


# ============================================================================
# Custom Strategies for generating test data
# ============================================================================

# Strategy for generating policy document names
policy_doc_name_strategy = st.sampled_from([
    "科技创新政策.txt", "乡村振兴战略.txt", "数字经济发展政策.txt",
    "环境保护条例.txt", "教育改革方案.txt", "医疗卫生规划.txt",
    "产业升级指导意见.txt", "人才引进办法.txt", "财政支持政策.txt",
    "区域协调发展规划.txt", "创新驱动发展战略.txt", "绿色发展行动计划.txt"
])

# Strategy for generating cited document names (used in citations)
cited_doc_name_strategy = st.sampled_from([
    "国家创新驱动发展战略纲要", "中华人民共和国科学技术进步法",
    "关于深化科技体制改革的若干意见", "国家中长期科学和技术发展规划纲要",
    "乡村振兴战略规划", "数字中国建设整体布局规划",
    "关于加快建设全国统一大市场的意见", "十四五规划纲要",
    "关于完善科技成果评价机制的指导意见", "国务院关于印发xxx的通知"
])


@st.composite
def raw_text_with_citations_strategy(draw, cited_docs: list = None):
    """
    Generate raw text containing citations to other documents.
    
    Args:
        cited_docs: Optional list of documents to cite
    
    Returns:
        str: Raw text with embedded citations
    """
    # Generate some base text
    base_phrases = [
        "为贯彻落实党中央、国务院决策部署，",
        "根据有关法律法规，",
        "按照相关要求，",
        "为推进高质量发展，",
        "结合本地实际情况，",
        "经研究决定，",
        "现就有关事项通知如下：",
        "加强组织领导，",
        "完善工作机制，",
        "强化监督检查，"
    ]
    
    # Decide how many citations to include
    num_citations = draw(st.integers(min_value=0, max_value=5))
    
    text_parts = []
    
    # Add some base text
    num_base = draw(st.integers(min_value=1, max_value=4))
    for _ in range(num_base):
        text_parts.append(draw(st.sampled_from(base_phrases)))
    
    # Add citations
    if cited_docs is None:
        cited_docs = [draw(cited_doc_name_strategy) for _ in range(num_citations)]
    
    for cited_doc in cited_docs[:num_citations]:
        citation_format = draw(st.sampled_from([
            f"根据《{cited_doc}》的规定，",
            f"依据《{cited_doc}》，",
            f"按照《{cited_doc}》要求，",
            f"贯彻《{cited_doc}》精神，",
            f"落实《{cited_doc}》，"
        ]))
        text_parts.append(citation_format)
        text_parts.append(draw(st.sampled_from(base_phrases)))
    
    return "".join(text_parts)


@st.composite
def documents_with_citations_strategy(draw):
    """
    Generate multiple documents with citation relationships.
    
    Returns:
        Tuple[List[str], List[str]]: (raw_texts, file_names)
    """
    num_docs = draw(st.integers(min_value=2, max_value=6))
    
    # Generate unique file names
    file_names = []
    for i in range(num_docs):
        name = f"政策文件{i+1}.txt"
        file_names.append(name)
    
    # Generate raw texts with citations
    raw_texts = []
    for i in range(num_docs):
        # Each document may cite some external documents
        text = draw(raw_text_with_citations_strategy())
        raw_texts.append(text)
    
    return raw_texts, file_names


@st.composite
def documents_with_known_citations_strategy(draw):
    """
    Generate documents with known citation structure for precise testing.
    
    Returns:
        Tuple[List[str], List[str], Dict[str, List[str]]]: 
            (raw_texts, file_names, expected_citations)
    """
    num_docs = draw(st.integers(min_value=2, max_value=4))
    
    # Generate file names
    file_names = [f"文档{i+1}.txt" for i in range(num_docs)]
    
    # Define some external documents that can be cited
    external_docs = [
        "国家创新驱动发展战略纲要",
        "十四五规划纲要",
        "乡村振兴战略规划"
    ]
    
    raw_texts = []
    expected_citations = {}
    
    for i, file_name in enumerate(file_names):
        # Decide which external docs this document will cite
        num_to_cite = draw(st.integers(min_value=0, max_value=len(external_docs)))
        docs_to_cite = draw(st.permutations(external_docs))[:num_to_cite]
        
        # Build text with these citations
        text_parts = ["为贯彻落实相关政策，"]
        for cited_doc in docs_to_cite:
            text_parts.append(f"根据《{cited_doc}》的规定，加强工作落实。")
        text_parts.append("特制定本方案。")
        
        raw_texts.append("".join(text_parts))
        expected_citations[file_name] = list(docs_to_cite)
    
    return raw_texts, file_names, expected_citations


# ============================================================================
# Property Tests
# ============================================================================

class TestCitationNetworkConsistencyProperty:
    """
    Property 9: 引用网络一致性
    
    *For any* 引用网络，文档的被引用次数应等于网络中指向该文档的入边数量，
    引用次数应等于出边数量。
    **Validates: Requirements 6.3, 6.5**
    """

    @given(data=documents_with_citations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_cited_by_count_equals_incoming_edges(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 9: 引用网络一致性**
        **Validates: Requirements 6.3, 6.5**
        
        Property: For any document in the citation network, its cited_by_count 
        should equal the number of incoming edges (edges pointing to it).
        """
        raw_texts, file_names = data
        
        analyzer = CitationAnalyzer(raw_texts, file_names)
        analyzer.extract_citations()
        
        # Build the network
        network = analyzer.build_citation_network()
        edges = network['edges']
        
        # Count incoming edges for each node
        incoming_edge_counts = Counter()
        for source, target in edges:
            incoming_edge_counts[target] += 1
        
        # Get citation stats
        stats = analyzer.get_all_citation_stats()
        
        # Verify cited_by_count equals incoming edges for each document
        for stat in stats:
            expected_incoming = incoming_edge_counts.get(stat.doc_name, 0)
            assert stat.cited_by_count == expected_incoming, (
                f"Document '{stat.doc_name}': cited_by_count ({stat.cited_by_count}) "
                f"!= incoming edges ({expected_incoming})"
            )

    @given(data=documents_with_citations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_cites_count_equals_outgoing_edges(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 9: 引用网络一致性**
        **Validates: Requirements 6.3, 6.5**
        
        Property: For any document in the citation network, its cites_count 
        should equal the number of outgoing edges (edges from it).
        """
        raw_texts, file_names = data
        
        analyzer = CitationAnalyzer(raw_texts, file_names)
        analyzer.extract_citations()
        
        # Build the network
        network = analyzer.build_citation_network()
        edges = network['edges']
        
        # Count outgoing edges for each node
        outgoing_edge_counts = Counter()
        for source, target in edges:
            outgoing_edge_counts[source] += 1
        
        # Get citation stats
        stats = analyzer.get_all_citation_stats()
        
        # Verify cites_count equals outgoing edges for each document
        for stat in stats:
            expected_outgoing = outgoing_edge_counts.get(stat.doc_name, 0)
            assert stat.cites_count == expected_outgoing, (
                f"Document '{stat.doc_name}': cites_count ({stat.cites_count}) "
                f"!= outgoing edges ({expected_outgoing})"
            )

    @given(data=documents_with_citations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_get_citation_count_consistency(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 9: 引用网络一致性**
        **Validates: Requirements 6.3, 6.5**
        
        Property: The get_citation_count method should return values consistent
        with the network edge counts.
        """
        raw_texts, file_names = data
        
        analyzer = CitationAnalyzer(raw_texts, file_names)
        analyzer.extract_citations()
        
        # Build the network
        network = analyzer.build_citation_network()
        edges = network['edges']
        
        # Count edges
        incoming_counts = Counter()
        outgoing_counts = Counter()
        for source, target in edges:
            incoming_counts[target] += 1
            outgoing_counts[source] += 1
        
        # Verify get_citation_count for each source document
        for file_name in file_names:
            cited_by, cites = analyzer.get_citation_count(file_name)
            
            expected_outgoing = outgoing_counts.get(file_name, 0)
            assert cites == expected_outgoing, (
                f"Document '{file_name}': get_citation_count cites ({cites}) "
                f"!= outgoing edges ({expected_outgoing})"
            )

    @given(data=documents_with_citations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_network_edges_match_citation_map(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 9: 引用网络一致性**
        **Validates: Requirements 6.3, 6.5**
        
        Property: The edges in the built network should match the citation map
        returned by extract_citations.
        """
        raw_texts, file_names = data
        
        analyzer = CitationAnalyzer(raw_texts, file_names)
        citation_map = analyzer.extract_citations()
        
        # Build the network
        network = analyzer.build_citation_network()
        edges = network['edges']
        
        # Convert edges to a set for comparison
        edge_set = set(edges)
        
        # Build expected edges from citation map
        expected_edges = set()
        for source_doc, cited_docs in citation_map.items():
            for cited_doc in cited_docs:
                expected_edges.add((source_doc, cited_doc))
        
        assert edge_set == expected_edges, (
            f"Network edges don't match citation map.\n"
            f"Network edges: {edge_set}\n"
            f"Expected from map: {expected_edges}"
        )

    @given(data=documents_with_citations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_all_cited_documents_in_network_nodes(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 9: 引用网络一致性**
        **Validates: Requirements 6.3, 6.5**
        
        Property: All cited documents should appear as nodes in the network.
        """
        raw_texts, file_names = data
        
        analyzer = CitationAnalyzer(raw_texts, file_names)
        citation_map = analyzer.extract_citations()
        
        # Build the network
        network = analyzer.build_citation_network()
        nodes = set(network['nodes'])
        
        # Collect all cited documents
        all_cited = set()
        for cited_docs in citation_map.values():
            all_cited.update(cited_docs)
        
        # Verify all cited documents are in nodes
        for cited_doc in all_cited:
            assert cited_doc in nodes, (
                f"Cited document '{cited_doc}' not found in network nodes"
            )

    @given(data=documents_with_citations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_all_source_documents_in_network_nodes(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 9: 引用网络一致性**
        **Validates: Requirements 6.3, 6.5**
        
        Property: All source documents (file_names) should appear as nodes in the network.
        """
        raw_texts, file_names = data
        
        analyzer = CitationAnalyzer(raw_texts, file_names)
        analyzer.extract_citations()
        
        # Build the network
        network = analyzer.build_citation_network()
        nodes = set(network['nodes'])
        
        # Verify all source documents are in nodes
        for file_name in file_names:
            assert file_name in nodes, (
                f"Source document '{file_name}' not found in network nodes"
            )

    @given(data=documents_with_citations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_citation_stats_cited_by_list_consistency(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 9: 引用网络一致性**
        **Validates: Requirements 6.3, 6.5**
        
        Property: The cited_by list in CitationStats should contain exactly
        the documents that have edges pointing to this document.
        """
        raw_texts, file_names = data
        
        analyzer = CitationAnalyzer(raw_texts, file_names)
        analyzer.extract_citations()
        
        # Build the network
        network = analyzer.build_citation_network()
        edges = network['edges']
        
        # Build incoming edge map
        incoming_sources = {}
        for source, target in edges:
            if target not in incoming_sources:
                incoming_sources[target] = set()
            incoming_sources[target].add(source)
        
        # Get citation stats
        stats = analyzer.get_all_citation_stats()
        
        # Verify cited_by list matches incoming edges
        for stat in stats:
            expected_sources = incoming_sources.get(stat.doc_name, set())
            actual_sources = set(stat.cited_by)
            
            assert actual_sources == expected_sources, (
                f"Document '{stat.doc_name}': cited_by list {actual_sources} "
                f"!= expected sources {expected_sources}"
            )

    @given(data=documents_with_citations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_citation_stats_cites_list_consistency(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 9: 引用网络一致性**
        **Validates: Requirements 6.3, 6.5**
        
        Property: The cites list in CitationStats should contain exactly
        the documents that this document has edges pointing to.
        """
        raw_texts, file_names = data
        
        analyzer = CitationAnalyzer(raw_texts, file_names)
        citation_map = analyzer.extract_citations()
        
        # Get citation stats
        stats = analyzer.get_all_citation_stats()
        
        # Verify cites list matches citation map
        for stat in stats:
            expected_cites = set(citation_map.get(stat.doc_name, []))
            actual_cites = set(stat.cites)
            
            assert actual_cites == expected_cites, (
                f"Document '{stat.doc_name}': cites list {actual_cites} "
                f"!= expected cites {expected_cites}"
            )


class TestCoreDocumentSortingProperty:
    """
    Property 10: 核心文档排序
    
    *For any* 引用网络中识别的核心文档列表，列表应按被引用次数降序排列。
    **Validates: Requirements 6.6**
    """

    @given(data=documents_with_citations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_core_documents_sorted_descending(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 10: 核心文档排序**
        **Validates: Requirements 6.6**
        
        Property: For any citation network, the core documents list returned by
        find_core_documents should be sorted in descending order by citation count.
        """
        raw_texts, file_names = data
        
        analyzer = CitationAnalyzer(raw_texts, file_names)
        analyzer.extract_citations()
        
        # Get core documents with various top_n values
        for top_n in [3, 5, 10, 20]:
            core_docs = analyzer.find_core_documents(top_n=top_n)
            
            # Skip if no core documents found
            if len(core_docs) < 2:
                continue
            
            # Verify descending order
            for i in range(len(core_docs) - 1):
                current_count = core_docs[i][1]
                next_count = core_docs[i + 1][1]
                
                assert current_count >= next_count, (
                    f"Core documents not sorted in descending order: "
                    f"'{core_docs[i][0]}' ({current_count}) should have >= citations "
                    f"than '{core_docs[i + 1][0]}' ({next_count})"
                )

    @given(data=documents_with_citations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_core_documents_citation_counts_match_network(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 10: 核心文档排序**
        **Validates: Requirements 6.6**
        
        Property: The citation counts in the core documents list should match
        the actual incoming edge counts in the network.
        """
        raw_texts, file_names = data
        
        analyzer = CitationAnalyzer(raw_texts, file_names)
        analyzer.extract_citations()
        
        # Build the network
        network = analyzer.build_citation_network()
        edges = network['edges']
        
        # Count incoming edges for each node
        incoming_counts = Counter()
        for source, target in edges:
            incoming_counts[target] += 1
        
        # Get core documents
        core_docs = analyzer.find_core_documents(top_n=20)
        
        # Verify citation counts match network
        for doc_name, count in core_docs:
            expected_count = incoming_counts.get(doc_name, 0)
            assert count == expected_count, (
                f"Core document '{doc_name}': reported count ({count}) "
                f"!= actual incoming edges ({expected_count})"
            )

    @given(data=documents_with_citations_strategy(), top_n=st.integers(min_value=1, max_value=20))
    @settings(max_examples=100, deadline=None)
    def test_core_documents_respects_top_n_limit(self, data: tuple, top_n: int):
        """
        **Feature: text-processing-enhancement, Property 10: 核心文档排序**
        **Validates: Requirements 6.6**
        
        Property: The find_core_documents method should return at most top_n documents.
        """
        raw_texts, file_names = data
        
        analyzer = CitationAnalyzer(raw_texts, file_names)
        analyzer.extract_citations()
        
        core_docs = analyzer.find_core_documents(top_n=top_n)
        
        assert len(core_docs) <= top_n, (
            f"find_core_documents returned {len(core_docs)} documents, "
            f"but top_n was {top_n}"
        )

    @given(data=documents_with_citations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_core_documents_contains_highest_cited(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 10: 核心文档排序**
        **Validates: Requirements 6.6**
        
        Property: If there are cited documents, the core documents list should
        contain the document(s) with the highest citation count.
        """
        raw_texts, file_names = data
        
        analyzer = CitationAnalyzer(raw_texts, file_names)
        analyzer.extract_citations()
        
        # Build the network
        network = analyzer.build_citation_network()
        edges = network['edges']
        
        # Skip if no edges
        if not edges:
            return
        
        # Count incoming edges for each node
        incoming_counts = Counter()
        for source, target in edges:
            incoming_counts[target] += 1
        
        # Find the maximum citation count
        max_count = max(incoming_counts.values())
        
        # Get core documents
        core_docs = analyzer.find_core_documents(top_n=20)
        
        # Skip if no core documents
        if not core_docs:
            return
        
        # The first document should have the maximum count
        first_doc_count = core_docs[0][1]
        assert first_doc_count == max_count, (
            f"First core document has count {first_doc_count}, "
            f"but maximum count in network is {max_count}"
        )

    @given(data=documents_with_citations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_core_documents_no_duplicates(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 10: 核心文档排序**
        **Validates: Requirements 6.6**
        
        Property: The core documents list should not contain duplicate entries.
        """
        raw_texts, file_names = data
        
        analyzer = CitationAnalyzer(raw_texts, file_names)
        analyzer.extract_citations()
        
        core_docs = analyzer.find_core_documents(top_n=20)
        
        # Extract document names
        doc_names = [doc[0] for doc in core_docs]
        
        # Check for duplicates
        assert len(doc_names) == len(set(doc_names)), (
            f"Core documents list contains duplicates: {doc_names}"
        )

    @given(data=documents_with_citations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_core_documents_all_have_positive_citations(self, data: tuple):
        """
        **Feature: text-processing-enhancement, Property 10: 核心文档排序**
        **Validates: Requirements 6.6**
        
        Property: All documents in the core documents list should have at least
        one citation (positive citation count).
        """
        raw_texts, file_names = data
        
        analyzer = CitationAnalyzer(raw_texts, file_names)
        analyzer.extract_citations()
        
        core_docs = analyzer.find_core_documents(top_n=20)
        
        # All core documents should have positive citation counts
        for doc_name, count in core_docs:
            assert count > 0, (
                f"Core document '{doc_name}' has zero citations, "
                f"should not be in core documents list"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
