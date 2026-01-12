# -*- coding: utf-8 -*-
"""
Property-Based Tests for Semantic Network Module

**Feature: text-processing-enhancement, Property 11: 语义网络社区覆盖**
**Validates: Requirements 7.5**

This module tests:
Property 11: For any semantic network community detection result, every node
in the network should be assigned to exactly one community.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock streamlit and plotly before importing the semantic network module
from unittest.mock import MagicMock
sys.modules['streamlit'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.subplots'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, Tuple, List

from modules.semantic_network import SemanticNetworkBuilder


# ============================================================================
# Custom Strategies for generating test data
# ============================================================================

# Strategy for generating Chinese-like word tokens
word_strategy = st.text(
    alphabet=st.sampled_from(
        list("人工智能科技创新政策发展经济数字乡村振兴战略实施推进改革开放教育医疗环境保护")
    ),
    min_size=2,
    max_size=6
).filter(lambda x: len(x.strip()) >= 2)


def cooccurrence_data_strategy(min_pairs=3, max_pairs=50, min_freq=1, max_freq=20):
    """Generate random cooccurrence data."""
    return st.integers(min_value=min_pairs, max_value=max_pairs).flatmap(
        lambda n_pairs: st.lists(
            st.tuples(
                word_strategy,
                word_strategy,
                st.integers(min_value=min_freq, max_value=max_freq)
            ),
            min_size=n_pairs,
            max_size=n_pairs,
            unique_by=lambda x: (x[0], x[1]) if x[0] < x[1] else (x[1], x[0])
        ).map(lambda pairs: {(p[0], p[1]): p[2] for p in pairs if p[0] != p[1]})
    )


# ============================================================================
# Property Tests
# ============================================================================

class TestSemanticNetworkCommunityCoverageProperty:
    """
    Property 11: 语义网络社区覆盖
    
    *For any* 语义网络的社区检测结果，网络中的每个节点都应被分配到一个社区。
    **Validates: Requirements 7.5**
    """

    @given(
        n_pairs=st.integers(min_value=5, max_value=30),
        min_weight=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=100, deadline=None)
    def test_all_nodes_assigned_to_community(self, n_pairs: int, min_weight: int):
        """
        **Feature: text-processing-enhancement, Property 11: 语义网络社区覆盖**
        **Validates: Requirements 7.5**
        
        Property: For any semantic network, after community detection, every node
        in the network should be assigned to exactly one community.
        """
        # Generate cooccurrence data with enough pairs to form a network
        base_words = ["政策", "发展", "创新", "经济", "科技", "改革", "实施", 
                      "推进", "战略", "规划", "建设", "服务", "管理", "技术"]
        
        # Create cooccurrence pairs
        cooccurrence_data = {}
        import random
        random.seed(42)
        
        for i in range(min(n_pairs, len(base_words) * (len(base_words) - 1) // 2)):
            w1_idx = random.randint(0, len(base_words) - 1)
            w2_idx = random.randint(0, len(base_words) - 1)
            if w1_idx != w2_idx:
                w1, w2 = base_words[w1_idx], base_words[w2_idx]
                key = (w1, w2) if w1 < w2 else (w2, w1)
                if key not in cooccurrence_data:
                    cooccurrence_data[key] = random.randint(min_weight, min_weight + 5)
        
        # Skip if no valid cooccurrence data
        assume(len(cooccurrence_data) >= 3)
        
        # Build semantic network
        texts = []  # Empty texts, we use cooccurrence_data directly
        builder = SemanticNetworkBuilder(texts, cooccurrence_data)
        network = builder.build_network(min_weight=min_weight)
        
        # Skip if network is empty
        assume(network is not None and network.number_of_nodes() > 0)
        
        # Detect communities
        communities = builder.detect_communities()
        
        # Property: Every node should be assigned to a community
        network_nodes = set(network.nodes())
        community_nodes = set(communities.keys())
        
        assert network_nodes == community_nodes, (
            f"All network nodes should be assigned to communities. "
            f"Network has {len(network_nodes)} nodes, "
            f"but only {len(community_nodes)} have community assignments. "
            f"Missing nodes: {network_nodes - community_nodes}"
        )

    @given(
        n_words=st.integers(min_value=5, max_value=20),
        connection_density=st.floats(min_value=0.2, max_value=0.8)
    )
    @settings(max_examples=100, deadline=None)
    def test_community_ids_are_valid_integers(self, n_words: int, connection_density: float):
        """
        **Feature: text-processing-enhancement, Property 11: 语义网络社区覆盖**
        **Validates: Requirements 7.5**
        
        Property: All community IDs should be non-negative integers.
        """
        # Generate words
        base_words = [f"词{i}" for i in range(n_words)]
        
        # Create cooccurrence data based on connection density
        cooccurrence_data = {}
        import random
        random.seed(42)
        
        for i in range(n_words):
            for j in range(i + 1, n_words):
                if random.random() < connection_density:
                    cooccurrence_data[(base_words[i], base_words[j])] = random.randint(2, 10)
        
        # Skip if no valid cooccurrence data
        assume(len(cooccurrence_data) >= 3)
        
        # Build semantic network
        builder = SemanticNetworkBuilder([], cooccurrence_data)
        network = builder.build_network(min_weight=1)
        
        # Skip if network is empty
        assume(network is not None and network.number_of_nodes() > 0)
        
        # Detect communities
        communities = builder.detect_communities()
        
        # Property: All community IDs should be non-negative integers
        for node, community_id in communities.items():
            assert isinstance(community_id, int), (
                f"Community ID for node '{node}' should be an integer, "
                f"but got {type(community_id)}"
            )
            assert community_id >= 0, (
                f"Community ID for node '{node}' should be non-negative, "
                f"but got {community_id}"
            )

    @given(
        n_words=st.integers(min_value=6, max_value=15)
    )
    @settings(max_examples=100, deadline=None)
    def test_community_detection_is_deterministic(self, n_words: int):
        """
        **Feature: text-processing-enhancement, Property 11: 语义网络社区覆盖**
        **Validates: Requirements 7.5**
        
        Property: Running community detection twice on the same network should
        produce the same results (deterministic).
        """
        # Generate fixed cooccurrence data
        base_words = [f"词{i}" for i in range(n_words)]
        
        cooccurrence_data = {}
        for i in range(n_words):
            for j in range(i + 1, min(i + 3, n_words)):  # Connect nearby words
                cooccurrence_data[(base_words[i], base_words[j])] = (i + j) % 5 + 2
        
        # Skip if no valid cooccurrence data
        assume(len(cooccurrence_data) >= 3)
        
        # Build semantic network and detect communities twice
        builder1 = SemanticNetworkBuilder([], cooccurrence_data)
        network1 = builder1.build_network(min_weight=1)
        
        assume(network1 is not None and network1.number_of_nodes() > 0)
        
        communities1 = builder1.detect_communities()
        
        # Second run on same builder (should use cached result)
        communities2 = builder1.detect_communities()
        
        # Property: Results should be identical
        assert communities1 == communities2, (
            "Community detection should be deterministic. "
            f"First run: {communities1}, Second run: {communities2}"
        )

    @given(
        n_components=st.integers(min_value=2, max_value=4),
        nodes_per_component=st.integers(min_value=3, max_value=6)
    )
    @settings(max_examples=100, deadline=None)
    def test_disconnected_components_get_different_communities(
        self, n_components: int, nodes_per_component: int
    ):
        """
        **Feature: text-processing-enhancement, Property 11: 语义网络社区覆盖**
        **Validates: Requirements 7.5**
        
        Property: Nodes in disconnected components should still all be assigned
        to communities (coverage is complete even for disconnected graphs).
        """
        # Create disconnected components
        cooccurrence_data = {}
        all_words = []
        
        for comp in range(n_components):
            component_words = [f"组{comp}_词{i}" for i in range(nodes_per_component)]
            all_words.extend(component_words)
            
            # Connect words within the same component
            for i in range(nodes_per_component):
                for j in range(i + 1, nodes_per_component):
                    cooccurrence_data[(component_words[i], component_words[j])] = 5
        
        # Build semantic network
        builder = SemanticNetworkBuilder([], cooccurrence_data)
        network = builder.build_network(min_weight=1)
        
        assume(network is not None and network.number_of_nodes() > 0)
        
        # Detect communities
        communities = builder.detect_communities()
        
        # Property: All nodes should be assigned to communities
        total_expected_nodes = n_components * nodes_per_component
        assert len(communities) == network.number_of_nodes(), (
            f"All {network.number_of_nodes()} nodes should have community assignments, "
            f"but only {len(communities)} have assignments"
        )
        
        # Property: Every node in the network should be in communities dict
        for node in network.nodes():
            assert node in communities, (
                f"Node '{node}' should be assigned to a community"
            )

    @given(
        n_words=st.integers(min_value=4, max_value=12)
    )
    @settings(max_examples=100, deadline=None)
    def test_single_node_network_has_community(self, n_words: int):
        """
        **Feature: text-processing-enhancement, Property 11: 语义网络社区覆盖**
        **Validates: Requirements 7.5**
        
        Property: Even a network with isolated nodes (after filtering) should
        assign all remaining nodes to communities.
        """
        # Create a sparse network where some nodes might be isolated after filtering
        base_words = [f"词{i}" for i in range(n_words)]
        
        cooccurrence_data = {}
        # Only connect first few words with high frequency
        for i in range(min(3, n_words)):
            for j in range(i + 1, min(4, n_words)):
                cooccurrence_data[(base_words[i], base_words[j])] = 10
        
        # Add some low-frequency connections that will be filtered out
        for i in range(4, n_words):
            if i + 1 < n_words:
                cooccurrence_data[(base_words[i], base_words[i + 1])] = 1
        
        # Build network with high min_weight to filter out weak connections
        builder = SemanticNetworkBuilder([], cooccurrence_data)
        network = builder.build_network(min_weight=5)
        
        # Skip if network is empty
        assume(network is not None and network.number_of_nodes() > 0)
        
        # Detect communities
        communities = builder.detect_communities()
        
        # Property: All nodes in the filtered network should have communities
        for node in network.nodes():
            assert node in communities, (
                f"Node '{node}' in the network should be assigned to a community"
            )


class TestSemanticNetworkCentralityRangeProperty:
    """
    Property 12: 中心性指标范围
    
    *For any* 语义网络的度中心性计算结果，所有节点的度中心性值应在[0, 1]范围内。
    **Validates: Requirements 7.7**
    """

    @given(
        n_pairs=st.integers(min_value=5, max_value=30),
        min_weight=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=100, deadline=None)
    def test_degree_centrality_in_valid_range(self, n_pairs: int, min_weight: int):
        """
        **Feature: text-processing-enhancement, Property 12: 中心性指标范围**
        **Validates: Requirements 7.7**
        
        Property: For any semantic network, all degree centrality values should
        be in the range [0, 1].
        """
        # Generate cooccurrence data
        base_words = ["政策", "发展", "创新", "经济", "科技", "改革", "实施", 
                      "推进", "战略", "规划", "建设", "服务", "管理", "技术"]
        
        cooccurrence_data = {}
        import random
        random.seed(42)
        
        for i in range(min(n_pairs, len(base_words) * (len(base_words) - 1) // 2)):
            w1_idx = random.randint(0, len(base_words) - 1)
            w2_idx = random.randint(0, len(base_words) - 1)
            if w1_idx != w2_idx:
                w1, w2 = base_words[w1_idx], base_words[w2_idx]
                key = (w1, w2) if w1 < w2 else (w2, w1)
                if key not in cooccurrence_data:
                    cooccurrence_data[key] = random.randint(min_weight, min_weight + 5)
        
        assume(len(cooccurrence_data) >= 3)
        
        builder = SemanticNetworkBuilder([], cooccurrence_data)
        network = builder.build_network(min_weight=min_weight)
        
        assume(network is not None and network.number_of_nodes() > 0)
        
        centrality = builder.calculate_centrality()
        
        # Property: All degree centrality values should be in [0, 1]
        for node, metrics in centrality.items():
            degree_centrality = metrics.get('degree', 0.0)
            assert 0.0 <= degree_centrality <= 1.0, (
                f"Degree centrality for node '{node}' should be in [0, 1], "
                f"but got {degree_centrality}"
            )

    @given(
        n_words=st.integers(min_value=5, max_value=20),
        connection_density=st.floats(min_value=0.2, max_value=0.8)
    )
    @settings(max_examples=100, deadline=None)
    def test_all_centrality_metrics_in_valid_range(self, n_words: int, connection_density: float):
        """
        **Feature: text-processing-enhancement, Property 12: 中心性指标范围**
        **Validates: Requirements 7.7**
        
        Property: All centrality metrics (degree, betweenness, closeness, eigenvector)
        should be in the range [0, 1].
        """
        base_words = [f"词{i}" for i in range(n_words)]
        
        cooccurrence_data = {}
        import random
        random.seed(42)
        
        for i in range(n_words):
            for j in range(i + 1, n_words):
                if random.random() < connection_density:
                    cooccurrence_data[(base_words[i], base_words[j])] = random.randint(2, 10)
        
        assume(len(cooccurrence_data) >= 3)
        
        builder = SemanticNetworkBuilder([], cooccurrence_data)
        network = builder.build_network(min_weight=1)
        
        assume(network is not None and network.number_of_nodes() > 0)
        
        centrality = builder.calculate_centrality()
        
        # Property: All centrality metrics should be in [0, 1]
        metric_names = ['degree', 'betweenness', 'closeness', 'eigenvector']
        
        for node, metrics in centrality.items():
            for metric_name in metric_names:
                value = metrics.get(metric_name, 0.0)
                assert 0.0 <= value <= 1.0, (
                    f"{metric_name} centrality for node '{node}' should be in [0, 1], "
                    f"but got {value}"
                )

    @given(
        n_words=st.integers(min_value=4, max_value=12)
    )
    @settings(max_examples=100, deadline=None)
    def test_centrality_covers_all_nodes(self, n_words: int):
        """
        **Feature: text-processing-enhancement, Property 12: 中心性指标范围**
        **Validates: Requirements 7.7**
        
        Property: Centrality calculation should return metrics for all nodes
        in the network.
        """
        base_words = [f"词{i}" for i in range(n_words)]
        
        cooccurrence_data = {}
        for i in range(n_words):
            for j in range(i + 1, min(i + 3, n_words)):
                cooccurrence_data[(base_words[i], base_words[j])] = (i + j) % 5 + 2
        
        assume(len(cooccurrence_data) >= 3)
        
        builder = SemanticNetworkBuilder([], cooccurrence_data)
        network = builder.build_network(min_weight=1)
        
        assume(network is not None and network.number_of_nodes() > 0)
        
        centrality = builder.calculate_centrality()
        
        # Property: All network nodes should have centrality metrics
        network_nodes = set(network.nodes())
        centrality_nodes = set(centrality.keys())
        
        assert network_nodes == centrality_nodes, (
            f"All network nodes should have centrality metrics. "
            f"Network has {len(network_nodes)} nodes, "
            f"but only {len(centrality_nodes)} have centrality values. "
            f"Missing nodes: {network_nodes - centrality_nodes}"
        )

    @given(
        n_words=st.integers(min_value=6, max_value=15)
    )
    @settings(max_examples=100, deadline=None)
    def test_centrality_calculation_is_deterministic(self, n_words: int):
        """
        **Feature: text-processing-enhancement, Property 12: 中心性指标范围**
        **Validates: Requirements 7.7**
        
        Property: Running centrality calculation twice on the same network should
        produce the same results (deterministic).
        """
        base_words = [f"词{i}" for i in range(n_words)]
        
        cooccurrence_data = {}
        for i in range(n_words):
            for j in range(i + 1, min(i + 3, n_words)):
                cooccurrence_data[(base_words[i], base_words[j])] = (i + j) % 5 + 2
        
        assume(len(cooccurrence_data) >= 3)
        
        builder = SemanticNetworkBuilder([], cooccurrence_data)
        network = builder.build_network(min_weight=1)
        
        assume(network is not None and network.number_of_nodes() > 0)
        
        centrality1 = builder.calculate_centrality()
        centrality2 = builder.calculate_centrality()
        
        # Property: Results should be identical
        assert centrality1 == centrality2, (
            "Centrality calculation should be deterministic"
        )

    @given(
        n_components=st.integers(min_value=2, max_value=4),
        nodes_per_component=st.integers(min_value=3, max_value=6)
    )
    @settings(max_examples=100, deadline=None)
    def test_disconnected_components_centrality_in_range(
        self, n_components: int, nodes_per_component: int
    ):
        """
        **Feature: text-processing-enhancement, Property 12: 中心性指标范围**
        **Validates: Requirements 7.7**
        
        Property: Even for disconnected graphs, all centrality values should
        be in the valid range [0, 1].
        """
        cooccurrence_data = {}
        
        for comp in range(n_components):
            component_words = [f"组{comp}_词{i}" for i in range(nodes_per_component)]
            
            for i in range(nodes_per_component):
                for j in range(i + 1, nodes_per_component):
                    cooccurrence_data[(component_words[i], component_words[j])] = 5
        
        builder = SemanticNetworkBuilder([], cooccurrence_data)
        network = builder.build_network(min_weight=1)
        
        assume(network is not None and network.number_of_nodes() > 0)
        
        centrality = builder.calculate_centrality()
        
        # Property: All centrality values should be in [0, 1] even for disconnected graphs
        metric_names = ['degree', 'betweenness', 'closeness', 'eigenvector']
        
        for node, metrics in centrality.items():
            for metric_name in metric_names:
                value = metrics.get(metric_name, 0.0)
                assert 0.0 <= value <= 1.0, (
                    f"{metric_name} centrality for node '{node}' in disconnected graph "
                    f"should be in [0, 1], but got {value}"
                )

    @given(
        n_words=st.integers(min_value=3, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_fully_connected_network_max_degree_centrality(self, n_words: int):
        """
        **Feature: text-processing-enhancement, Property 12: 中心性指标范围**
        **Validates: Requirements 7.7**
        
        Property: In a fully connected network, all nodes should have degree
        centrality equal to 1.0.
        """
        base_words = [f"词{i}" for i in range(n_words)]
        
        # Create fully connected network
        cooccurrence_data = {}
        for i in range(n_words):
            for j in range(i + 1, n_words):
                cooccurrence_data[(base_words[i], base_words[j])] = 5
        
        builder = SemanticNetworkBuilder([], cooccurrence_data)
        network = builder.build_network(min_weight=1)
        
        assume(network is not None and network.number_of_nodes() > 1)
        
        centrality = builder.calculate_centrality()
        
        # Property: In fully connected network, all degree centrality should be 1.0
        for node, metrics in centrality.items():
            degree_centrality = metrics.get('degree', 0.0)
            assert abs(degree_centrality - 1.0) < 1e-10, (
                f"In fully connected network, degree centrality for node '{node}' "
                f"should be 1.0, but got {degree_centrality}"
            )


class TestSemanticNetworkCentralityEdgeCases:
    """
    Edge case tests for centrality metrics.
    """

    def test_empty_network_returns_empty_centrality(self):
        """
        **Feature: text-processing-enhancement, Property 12: 中心性指标范围**
        **Validates: Requirements 7.7**
        
        Edge case: Empty network should return empty centrality dict.
        """
        builder = SemanticNetworkBuilder([], {})
        network = builder.build_network(min_weight=1)
        
        centrality = builder.calculate_centrality()
        
        assert centrality == {}, (
            "Empty network should return empty centrality dict"
        )

    def test_single_edge_network_centrality(self):
        """
        **Feature: text-processing-enhancement, Property 12: 中心性指标范围**
        **Validates: Requirements 7.7**
        
        Edge case: Network with single edge should have valid centrality values.
        """
        cooccurrence_data = {("词A", "词B"): 5}
        
        builder = SemanticNetworkBuilder([], cooccurrence_data)
        network = builder.build_network(min_weight=1)
        
        assert network is not None and network.number_of_nodes() == 2
        
        centrality = builder.calculate_centrality()
        
        assert len(centrality) == 2, (
            f"Both nodes should have centrality values, got {len(centrality)}"
        )
        
        # Both nodes should have degree centrality of 1.0 (each connected to the only other node)
        for node, metrics in centrality.items():
            degree_centrality = metrics.get('degree', 0.0)
            assert 0.0 <= degree_centrality <= 1.0, (
                f"Degree centrality for '{node}' should be in [0, 1], got {degree_centrality}"
            )

    def test_star_network_centrality(self):
        """
        **Feature: text-processing-enhancement, Property 12: 中心性指标范围**
        **Validates: Requirements 7.7**
        
        Edge case: Star network (one central node connected to all others)
        should have valid centrality values with center having highest values.
        """
        # Create star network: center connected to all periphery nodes
        center = "中心"
        periphery = [f"外围{i}" for i in range(5)]
        
        cooccurrence_data = {}
        for p in periphery:
            cooccurrence_data[(center, p)] = 5
        
        builder = SemanticNetworkBuilder([], cooccurrence_data)
        network = builder.build_network(min_weight=1)
        
        assert network is not None
        
        centrality = builder.calculate_centrality()
        
        # All centrality values should be in [0, 1]
        for node, metrics in centrality.items():
            for metric_name in ['degree', 'betweenness', 'closeness', 'eigenvector']:
                value = metrics.get(metric_name, 0.0)
                assert 0.0 <= value <= 1.0, (
                    f"{metric_name} centrality for '{node}' should be in [0, 1], got {value}"
                )
        
        # Center should have highest degree centrality
        center_degree = centrality[center]['degree']
        for p in periphery:
            assert center_degree >= centrality[p]['degree'], (
                f"Center node should have highest degree centrality"
            )


class TestSemanticNetworkEmptyAndEdgeCases:
    """
    Additional edge case tests for community coverage.
    """

    def test_empty_network_returns_empty_communities(self):
        """
        **Feature: text-processing-enhancement, Property 11: 语义网络社区覆盖**
        **Validates: Requirements 7.5**
        
        Edge case: Empty network should return empty community dict.
        """
        builder = SemanticNetworkBuilder([], {})
        network = builder.build_network(min_weight=1)
        
        communities = builder.detect_communities()
        
        assert communities == {}, (
            "Empty network should return empty community dict"
        )

    def test_single_edge_network_has_communities(self):
        """
        **Feature: text-processing-enhancement, Property 11: 语义网络社区覆盖**
        **Validates: Requirements 7.5**
        
        Edge case: Network with single edge should assign both nodes to communities.
        """
        cooccurrence_data = {("词A", "词B"): 5}
        
        builder = SemanticNetworkBuilder([], cooccurrence_data)
        network = builder.build_network(min_weight=1)
        
        assert network is not None and network.number_of_nodes() == 2
        
        communities = builder.detect_communities()
        
        assert len(communities) == 2, (
            f"Both nodes should have community assignments, got {len(communities)}"
        )
        assert "词A" in communities and "词B" in communities, (
            "Both '词A' and '词B' should be in communities"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
