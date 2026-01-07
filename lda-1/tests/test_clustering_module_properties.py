# -*- coding: utf-8 -*-
"""
Property-Based Tests for Clustering Module

**Feature: text-processing-enhancement, Property 4: 聚类结果完整性**
**Validates: Requirements 3.2, 3.4**

**Feature: text-processing-enhancement, Property 5: 分类标签覆盖**
**Validates: Requirements 3.6**

This module tests:
1. Property 4: For any document collection and specified number of clusters K,
   the clustering result should assign all documents to K clusters, and each
   cluster should have representative keywords.
2. Property 5: For any partially labeled document collection, after automatic
   classification, all documents should have classification labels.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock streamlit and plotly before importing the clustering module
from unittest.mock import MagicMock
sys.modules['streamlit'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.subplots'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from modules.clustering_module import TextClusterer, TextClassifier, create_doc_vectors


# ============================================================================
# Custom Strategies for generating test data
# ============================================================================

# Strategy for generating document vectors (n_docs x n_features)
def doc_vectors_strategy(min_docs=2, max_docs=20, min_features=5, max_features=50):
    """Generate random document vectors."""
    return st.integers(min_value=min_docs, max_value=max_docs).flatmap(
        lambda n_docs: st.integers(min_value=min_features, max_value=max_features).flatmap(
            lambda n_features: st.lists(
                st.lists(
                    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=n_features,
                    max_size=n_features
                ),
                min_size=n_docs,
                max_size=n_docs
            ).map(lambda x: np.array(x))
        )
    )


# Strategy for generating file names
file_name_strategy = st.text(
    alphabet=st.sampled_from(list("abcdefghijklmnopqrstuvwxyz0123456789_政策文件")),
    min_size=1,
    max_size=20
).filter(lambda x: x.strip() != "").map(lambda x: f"{x}.txt")


# Strategy for generating tokenized texts (list of word lists)
word_strategy = st.text(
    alphabet=st.sampled_from(
        list("人工智能科技创新政策发展经济数字乡村振兴战略实施推进改革开放")
    ),
    min_size=1,
    max_size=10
).filter(lambda x: x.strip() != "")

tokenized_text_strategy = st.lists(
    word_strategy,
    min_size=5,
    max_size=50
)


# ============================================================================
# Property Tests
# ============================================================================

class TestClusteringResultCompletenessProperty:
    """
    Property 4: 聚类结果完整性
    
    *For any* 文档集合和指定的聚类数量K，聚类结果应将所有文档分配到K个聚类中，
    且每个聚类都有代表性关键词。
    **Validates: Requirements 3.2, 3.4**
    """

    @given(
        n_docs=st.integers(min_value=3, max_value=20),
        n_features=st.integers(min_value=5, max_value=30),
        n_clusters=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_kmeans_assigns_all_documents_to_clusters(
        self, n_docs: int, n_features: int, n_clusters: int
    ):
        """
        **Feature: text-processing-enhancement, Property 4: 聚类结果完整性**
        **Validates: Requirements 3.2, 3.4**
        
        Property: For any document collection and K clusters using K-means,
        all documents should be assigned to exactly one of the K clusters.
        """
        # Ensure n_clusters <= n_docs
        n_clusters = min(n_clusters, n_docs)
        assume(n_clusters >= 2)
        
        # Generate random document vectors
        np.random.seed(42)
        doc_vectors = np.random.rand(n_docs, n_features)
        
        # Generate file names
        file_names = [f"doc_{i}.txt" for i in range(n_docs)]
        
        # Create clusterer and perform clustering
        clusterer = TextClusterer(doc_vectors, file_names)
        cluster_labels = clusterer.kmeans_clustering(n_clusters)
        
        # Property 1: All documents should be assigned
        assert len(cluster_labels) == n_docs, (
            f"Number of cluster labels ({len(cluster_labels)}) should equal "
            f"number of documents ({n_docs})"
        )
        
        # Property 2: All labels should be valid cluster IDs (0 to n_clusters-1)
        unique_labels = set(cluster_labels)
        for label in cluster_labels:
            assert 0 <= label < n_clusters, (
                f"Cluster label {label} should be in range [0, {n_clusters})"
            )
        
        # Property 3: The number of unique clusters should be <= n_clusters
        assert len(unique_labels) <= n_clusters, (
            f"Number of unique clusters ({len(unique_labels)}) should be <= {n_clusters}"
        )

    @given(
        n_docs=st.integers(min_value=3, max_value=20),
        n_features=st.integers(min_value=5, max_value=30),
        n_clusters=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_hierarchical_assigns_all_documents_to_clusters(
        self, n_docs: int, n_features: int, n_clusters: int
    ):
        """
        **Feature: text-processing-enhancement, Property 4: 聚类结果完整性**
        **Validates: Requirements 3.2, 3.4**
        
        Property: For any document collection and K clusters using hierarchical
        clustering, all documents should be assigned to exactly one of the K clusters.
        """
        # Ensure n_clusters <= n_docs
        n_clusters = min(n_clusters, n_docs)
        assume(n_clusters >= 2)
        
        # Generate random document vectors
        np.random.seed(42)
        doc_vectors = np.random.rand(n_docs, n_features)
        
        # Generate file names
        file_names = [f"doc_{i}.txt" for i in range(n_docs)]
        
        # Create clusterer and perform clustering
        clusterer = TextClusterer(doc_vectors, file_names)
        cluster_labels = clusterer.hierarchical_clustering(n_clusters)
        
        # Property 1: All documents should be assigned
        assert len(cluster_labels) == n_docs, (
            f"Number of cluster labels ({len(cluster_labels)}) should equal "
            f"number of documents ({n_docs})"
        )
        
        # Property 2: All labels should be valid cluster IDs (0 to n_clusters-1)
        for label in cluster_labels:
            assert 0 <= label < n_clusters, (
                f"Cluster label {label} should be in range [0, {n_clusters})"
            )
        
        # Property 3: The number of unique clusters should be <= n_clusters
        unique_labels = set(cluster_labels)
        assert len(unique_labels) <= n_clusters, (
            f"Number of unique clusters ({len(unique_labels)}) should be <= {n_clusters}"
        )

    @given(
        n_docs=st.integers(min_value=3, max_value=15),
        n_clusters=st.integers(min_value=2, max_value=4),
        words_per_doc=st.integers(min_value=10, max_value=30)
    )
    @settings(max_examples=100, deadline=None)
    def test_each_cluster_has_keywords(
        self, n_docs: int, n_clusters: int, words_per_doc: int
    ):
        """
        **Feature: text-processing-enhancement, Property 4: 聚类结果完整性**
        **Validates: Requirements 3.4**
        
        Property: For any clustering result, each cluster that contains at least
        one document should have representative keywords.
        """
        # Ensure n_clusters <= n_docs
        n_clusters = min(n_clusters, n_docs)
        assume(n_clusters >= 2)
        
        # Generate random document vectors
        np.random.seed(42)
        n_features = 20
        doc_vectors = np.random.rand(n_docs, n_features)
        
        # Generate file names
        file_names = [f"doc_{i}.txt" for i in range(n_docs)]
        
        # Generate tokenized texts with some common words
        base_words = ["政策", "发展", "创新", "经济", "科技", "改革", "实施", "推进"]
        texts = []
        for i in range(n_docs):
            # Each document has some base words plus some unique words
            doc_words = base_words.copy()
            for j in range(words_per_doc - len(base_words)):
                doc_words.append(f"词{i}_{j}")
            np.random.shuffle(doc_words)
            texts.append(doc_words)
        
        # Create clusterer and perform clustering
        clusterer = TextClusterer(doc_vectors, file_names)
        cluster_labels = clusterer.kmeans_clustering(n_clusters)
        
        # Get keywords for all clusters
        all_keywords = clusterer.get_all_cluster_keywords(texts, top_n=10)
        
        # Property: Each cluster with documents should have keywords
        for cluster_id in range(n_clusters):
            cluster_docs = clusterer.get_cluster_documents(cluster_id)
            if len(cluster_docs) > 0:
                keywords = all_keywords.get(cluster_id, [])
                assert len(keywords) > 0, (
                    f"Cluster {cluster_id} has {len(cluster_docs)} documents "
                    f"but no keywords"
                )

    @given(
        n_docs=st.integers(min_value=3, max_value=15),
        n_clusters=st.integers(min_value=2, max_value=4)
    )
    @settings(max_examples=100, deadline=None)
    def test_cluster_documents_partition(
        self, n_docs: int, n_clusters: int
    ):
        """
        **Feature: text-processing-enhancement, Property 4: 聚类结果完整性**
        **Validates: Requirements 3.2**
        
        Property: The documents in all clusters should form a partition of the
        original document set (each document appears in exactly one cluster).
        """
        # Ensure n_clusters <= n_docs
        n_clusters = min(n_clusters, n_docs)
        assume(n_clusters >= 2)
        
        # Generate random document vectors
        np.random.seed(42)
        n_features = 20
        doc_vectors = np.random.rand(n_docs, n_features)
        
        # Generate file names
        file_names = [f"doc_{i}.txt" for i in range(n_docs)]
        
        # Create clusterer and perform clustering
        clusterer = TextClusterer(doc_vectors, file_names)
        clusterer.kmeans_clustering(n_clusters)
        
        # Collect all documents from all clusters
        all_clustered_docs = []
        for cluster_id in range(n_clusters):
            cluster_docs = clusterer.get_cluster_documents(cluster_id)
            all_clustered_docs.extend(cluster_docs)
        
        # Property 1: Total number of clustered documents equals original count
        assert len(all_clustered_docs) == n_docs, (
            f"Total clustered documents ({len(all_clustered_docs)}) should equal "
            f"original document count ({n_docs})"
        )
        
        # Property 2: No duplicates (each document in exactly one cluster)
        assert len(set(all_clustered_docs)) == len(all_clustered_docs), (
            "Documents should not appear in multiple clusters"
        )
        
        # Property 3: All original documents are present
        assert set(all_clustered_docs) == set(file_names), (
            "All original documents should be present in clustering result"
        )

    @given(
        n_docs=st.integers(min_value=3, max_value=15),
        n_clusters=st.integers(min_value=2, max_value=4)
    )
    @settings(max_examples=100, deadline=None)
    def test_clustering_export_contains_all_documents(
        self, n_docs: int, n_clusters: int
    ):
        """
        **Feature: text-processing-enhancement, Property 4: 聚类结果完整性**
        **Validates: Requirements 3.2**
        
        Property: The exported clustering results should contain all documents
        with valid cluster assignments.
        """
        # Ensure n_clusters <= n_docs
        n_clusters = min(n_clusters, n_docs)
        assume(n_clusters >= 2)
        
        # Generate random document vectors
        np.random.seed(42)
        n_features = 20
        doc_vectors = np.random.rand(n_docs, n_features)
        
        # Generate file names
        file_names = [f"doc_{i}.txt" for i in range(n_docs)]
        
        # Create clusterer and perform clustering
        clusterer = TextClusterer(doc_vectors, file_names)
        clusterer.kmeans_clustering(n_clusters)
        
        # Export results
        csv_content = clusterer.export_results()
        
        # Parse CSV content
        lines = csv_content.strip().split('\n')
        
        # Property 1: Should have header + n_docs data lines
        assert len(lines) == n_docs + 1, (
            f"Export should have {n_docs + 1} lines (header + data), "
            f"but has {len(lines)}"
        )
        
        # Property 2: Each document should appear exactly once
        exported_docs = []
        for line in lines[1:]:  # Skip header
            parts = line.split(',')
            if len(parts) >= 2:
                exported_docs.append(parts[0])
        
        assert set(exported_docs) == set(file_names), (
            "Exported documents should match original file names"
        )


class TestClassificationLabelCoverageProperty:
    """
    Property 5: 分类标签覆盖
    
    *For any* 部分标注的文档集合，自动分类后所有文档都应有分类标签。
    **Validates: Requirements 3.6**
    """

    @given(
        n_docs=st.integers(min_value=4, max_value=20),
        n_features=st.integers(min_value=5, max_value=30),
        n_labels=st.integers(min_value=2, max_value=4),
        labeled_ratio=st.floats(min_value=0.3, max_value=0.7)
    )
    @settings(max_examples=100, deadline=None)
    def test_all_documents_have_labels_after_classification(
        self, n_docs: int, n_features: int, n_labels: int, labeled_ratio: float
    ):
        """
        **Feature: text-processing-enhancement, Property 5: 分类标签覆盖**
        **Validates: Requirements 3.6**
        
        Property: For any partially labeled document collection, after automatic
        classification, all documents should have classification labels.
        """
        # Calculate number of labeled documents (at least 2 for training)
        n_labeled = max(2, int(n_docs * labeled_ratio))
        # Ensure we have at least some unlabeled documents
        assume(n_labeled < n_docs)
        # Ensure we have at least 2 different labels
        assume(n_labels >= 2)
        
        # Generate random document vectors
        np.random.seed(42)
        doc_vectors = np.random.rand(n_docs, n_features)
        
        # Generate file names
        file_names = [f"doc_{i}.txt" for i in range(n_docs)]
        
        # Create classifier
        classifier = TextClassifier(doc_vectors, file_names)
        
        # Add label categories
        labels = [f"类别{i}" for i in range(n_labels)]
        for label in labels:
            classifier.add_label_category(label)
        
        # Manually label some documents (ensuring at least 2 different labels)
        labeled_indices = list(range(n_labeled))
        for i, idx in enumerate(labeled_indices):
            # Distribute labels to ensure at least 2 different labels
            label = labels[i % n_labels]
            classifier.add_label(file_names[idx], label)
        
        # Train classifier
        train_success = classifier.train_classifier()
        assume(train_success)  # Skip if training fails
        
        # Predict unlabeled documents
        predictions = classifier.predict_unlabeled()
        
        # Get all labels (manual + predicted)
        all_labels = classifier.get_all_labels()
        
        # Property: All documents should have labels
        assert len(all_labels) == n_docs, (
            f"All {n_docs} documents should have labels, "
            f"but only {len(all_labels)} have labels"
        )
        
        # Property: All file names should be in the labels
        for name in file_names:
            assert name in all_labels, (
                f"Document '{name}' should have a label after classification"
            )

    @given(
        n_docs=st.integers(min_value=4, max_value=15),
        n_features=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_predicted_labels_are_valid(
        self, n_docs: int, n_features: int
    ):
        """
        **Feature: text-processing-enhancement, Property 5: 分类标签覆盖**
        **Validates: Requirements 3.6**
        
        Property: All predicted labels should be from the set of available labels.
        """
        # Generate random document vectors
        np.random.seed(42)
        doc_vectors = np.random.rand(n_docs, n_features)
        
        # Generate file names
        file_names = [f"doc_{i}.txt" for i in range(n_docs)]
        
        # Create classifier
        classifier = TextClassifier(doc_vectors, file_names)
        
        # Add label categories
        labels = ["政策类", "经济类", "科技类"]
        for label in labels:
            classifier.add_label_category(label)
        
        # Manually label first 3 documents with different labels
        classifier.add_label(file_names[0], "政策类")
        classifier.add_label(file_names[1], "经济类")
        classifier.add_label(file_names[2], "科技类")
        
        # Train classifier
        train_success = classifier.train_classifier()
        assume(train_success)
        
        # Predict unlabeled documents
        predictions = classifier.predict_unlabeled()
        
        # Property: All predicted labels should be valid
        for doc_name, predicted_label in predictions.items():
            assert predicted_label in labels, (
                f"Predicted label '{predicted_label}' for document '{doc_name}' "
                f"should be one of {labels}"
            )

    @given(
        n_docs=st.integers(min_value=4, max_value=15),
        n_features=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_manual_labels_preserved_after_classification(
        self, n_docs: int, n_features: int
    ):
        """
        **Feature: text-processing-enhancement, Property 5: 分类标签覆盖**
        **Validates: Requirements 3.6**
        
        Property: Manual labels should be preserved after automatic classification.
        """
        # Generate random document vectors
        np.random.seed(42)
        doc_vectors = np.random.rand(n_docs, n_features)
        
        # Generate file names
        file_names = [f"doc_{i}.txt" for i in range(n_docs)]
        
        # Create classifier
        classifier = TextClassifier(doc_vectors, file_names)
        
        # Add label categories
        labels = ["类别A", "类别B"]
        for label in labels:
            classifier.add_label_category(label)
        
        # Manually label first 2 documents
        manual_labels = {
            file_names[0]: "类别A",
            file_names[1]: "类别B"
        }
        for doc_name, label in manual_labels.items():
            classifier.add_label(doc_name, label)
        
        # Train classifier
        train_success = classifier.train_classifier()
        assume(train_success)
        
        # Predict unlabeled documents
        classifier.predict_unlabeled()
        
        # Get all labels
        all_labels = classifier.get_all_labels()
        
        # Property: Manual labels should be preserved
        for doc_name, expected_label in manual_labels.items():
            assert all_labels.get(doc_name) == expected_label, (
                f"Manual label for '{doc_name}' should be '{expected_label}', "
                f"but got '{all_labels.get(doc_name)}'"
            )

    @given(
        n_docs=st.integers(min_value=4, max_value=15),
        n_features=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_export_contains_all_documents_with_labels(
        self, n_docs: int, n_features: int
    ):
        """
        **Feature: text-processing-enhancement, Property 5: 分类标签覆盖**
        **Validates: Requirements 3.6**
        
        Property: Exported classification results should contain all documents
        with their labels.
        """
        # Generate random document vectors
        np.random.seed(42)
        doc_vectors = np.random.rand(n_docs, n_features)
        
        # Generate file names
        file_names = [f"doc_{i}.txt" for i in range(n_docs)]
        
        # Create classifier
        classifier = TextClassifier(doc_vectors, file_names)
        
        # Add label categories
        labels = ["类别A", "类别B"]
        for label in labels:
            classifier.add_label_category(label)
        
        # Manually label first 2 documents
        classifier.add_label(file_names[0], "类别A")
        classifier.add_label(file_names[1], "类别B")
        
        # Train classifier
        train_success = classifier.train_classifier()
        assume(train_success)
        
        # Predict unlabeled documents
        classifier.predict_unlabeled()
        
        # Export results
        csv_content = classifier.export_results()
        
        # Parse CSV content
        lines = csv_content.strip().split('\n')
        
        # Property 1: Should have header + n_docs data lines
        assert len(lines) == n_docs + 1, (
            f"Export should have {n_docs + 1} lines (header + data), "
            f"but has {len(lines)}"
        )
        
        # Property 2: Each document should appear exactly once
        exported_docs = []
        for line in lines[1:]:  # Skip header
            parts = line.split(',')
            if len(parts) >= 2:
                exported_docs.append(parts[0])
        
        assert set(exported_docs) == set(file_names), (
            "Exported documents should match original file names"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
