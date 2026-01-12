# -*- coding: utf-8 -*-
"""
Property-Based Tests for Qualitative Coding Module

**Feature: text-processing-enhancement, Property 1: 编码数据持久化一致性**
**Validates: Requirements 1.6, 1.7**

This module tests the property that for any coding scheme and coded segments,
saving to a file and then loading should result in equivalent data.
"""

import os
import sys
import tempfile
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from modules.qualitative_coding import (
    Code, CodedSegment, CodingScheme, QualitativeCoder
)


# ============================================================================
# Custom Strategies for generating test data
# ============================================================================

# Strategy for generating valid code names (non-empty, alphanumeric + Chinese)
code_name_strategy = st.text(
    alphabet=st.sampled_from(
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") +
        list("编码主题分类标签政策创新发展")
    ),
    min_size=1,
    max_size=30
).filter(lambda x: x.strip() != "")

# Strategy for generating code descriptions
description_strategy = st.text(
    alphabet=st.sampled_from(
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ") +
        list("这是一个编码描述用于测试持久化功能")
    ),
    min_size=0,
    max_size=100
)

# Strategy for generating valid hex colors
color_strategy = st.sampled_from([
    "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#34495e", "#e67e22", "#95a5a6", "#d35400"
])

# Strategy for generating document IDs
doc_id_strategy = st.text(
    alphabet=st.sampled_from(
        list("abcdefghijklmnopqrstuvwxyz0123456789_-.") +
        list("文档政策")
    ),
    min_size=1,
    max_size=50
).filter(lambda x: x.strip() != "")

# Strategy for generating text content
text_content_strategy = st.text(
    alphabet=st.sampled_from(
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ") +
        list("这是一段测试文本用于验证编码持久化功能政策创新发展")
    ),
    min_size=1,
    max_size=200
).filter(lambda x: x.strip() != "")

# Strategy for generating notes
note_strategy = st.text(
    alphabet=st.sampled_from(
        list("abcdefghijklmnopqrstuvwxyz ") +
        list("备注说明")
    ),
    min_size=0,
    max_size=100
)


# Strategy for generating a Code object
@st.composite
def code_strategy(draw):
    """Generate a valid Code object."""
    name = draw(code_name_strategy)
    description = draw(description_strategy)
    color = draw(color_strategy)
    return Code(name=name, description=description, color=color, parent=None)


# Strategy for generating a list of codes (for a coding scheme)
@st.composite
def codes_list_strategy(draw, min_size=0, max_size=10):
    """Generate a list of unique codes."""
    num_codes = draw(st.integers(min_value=min_size, max_value=max_size))
    codes = []
    used_names = set()
    
    for _ in range(num_codes):
        name = draw(code_name_strategy)
        # Ensure unique names
        while name in used_names:
            name = draw(code_name_strategy)
        used_names.add(name)
        
        description = draw(description_strategy)
        color = draw(color_strategy)
        codes.append(Code(name=name, description=description, color=color, parent=None))
    
    return codes


# Strategy for generating a coded segment
@st.composite
def coded_segment_strategy(draw, available_codes):
    """Generate a valid CodedSegment with codes from available_codes."""
    doc_id = draw(doc_id_strategy)
    text = draw(text_content_strategy)
    note = draw(note_strategy)
    
    # Generate valid positions
    text_len = len(text)
    start_pos = draw(st.integers(min_value=0, max_value=max(0, text_len - 1)))
    end_pos = draw(st.integers(min_value=start_pos + 1, max_value=text_len + 100))
    
    # Select codes from available codes
    if available_codes:
        num_codes = draw(st.integers(min_value=1, max_value=min(3, len(available_codes))))
        selected_codes = draw(st.lists(
            st.sampled_from(available_codes),
            min_size=num_codes,
            max_size=num_codes,
            unique=True
        ))
    else:
        selected_codes = []
    
    return CodedSegment(
        document_id=doc_id,
        start_pos=start_pos,
        end_pos=end_pos,
        text=text,
        codes=selected_codes,
        note=note
    )


# ============================================================================
# Property Tests
# ============================================================================

class TestCodingPersistenceProperty:
    """
    Property 1: 编码数据持久化一致性
    
    *For any* 编码方案和已编码片段集合，保存到文件后再加载，
    应该得到与原始数据等价的编码方案和片段集合。
    **Validates: Requirements 1.6, 1.7**
    """

    @given(
        scheme_name=code_name_strategy,
        scheme_description=description_strategy,
        codes=codes_list_strategy(min_size=0, max_size=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_coding_scheme_save_load_roundtrip(
        self, scheme_name: str, scheme_description: str, codes: list
    ):
        """
        **Feature: text-processing-enhancement, Property 1: 编码数据持久化一致性**
        **Validates: Requirements 1.6, 1.7**
        
        Property: For any coding scheme, saving to file and loading should
        result in an equivalent coding scheme with the same codes.
        """
        # Create a coding scheme
        scheme = CodingScheme(name=scheme_name, description=scheme_description)
        
        # Add codes to the scheme
        for code in codes:
            scheme.add_code(
                name=code.name,
                description=code.description,
                color=code.color,
                parent=code.parent
            )
        
        # Create a coder with the scheme
        coder = QualitativeCoder(scheme=scheme)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, encoding='utf-8'
        ) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save the scheme
            save_success = coder.save_scheme(tmp_path)
            assert save_success, "Save should succeed"
            
            # Create a new coder and load
            new_coder = QualitativeCoder()
            load_success = new_coder.load_scheme(tmp_path)
            assert load_success, "Load should succeed"
            
            # Verify scheme name and description
            assert new_coder.scheme.name == scheme_name, (
                f"Scheme name mismatch: expected={scheme_name}, "
                f"actual={new_coder.scheme.name}"
            )
            assert new_coder.scheme.description == scheme_description, (
                f"Scheme description mismatch: expected={scheme_description}, "
                f"actual={new_coder.scheme.description}"
            )
            
            # Verify code count
            assert new_coder.scheme.get_code_count() == len(codes), (
                f"Code count mismatch: expected={len(codes)}, "
                f"actual={new_coder.scheme.get_code_count()}"
            )
            
            # Verify each code
            for original_code in codes:
                loaded_code = new_coder.scheme.get_code(original_code.name)
                assert loaded_code is not None, (
                    f"Code '{original_code.name}' should exist after loading"
                )
                assert loaded_code.description == original_code.description, (
                    f"Code description mismatch for '{original_code.name}'"
                )
                assert loaded_code.color == original_code.color, (
                    f"Code color mismatch for '{original_code.name}'"
                )
                assert loaded_code.parent == original_code.parent, (
                    f"Code parent mismatch for '{original_code.name}'"
                )
            
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @given(
        scheme_name=code_name_strategy,
        codes=codes_list_strategy(min_size=1, max_size=5),
        num_segments=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_coded_segments_save_load_roundtrip(
        self, scheme_name: str, codes: list, num_segments: int
    ):
        """
        **Feature: text-processing-enhancement, Property 1: 编码数据持久化一致性**
        **Validates: Requirements 1.6, 1.7**
        
        Property: For any coded segments, saving to file and loading should
        result in equivalent segments with the same data.
        """
        assume(len(codes) >= 1)
        
        # Create a coding scheme with codes
        scheme = CodingScheme(name=scheme_name)
        for code in codes:
            scheme.add_code(
                name=code.name,
                description=code.description,
                color=code.color
            )
        
        # Create a coder with the scheme
        coder = QualitativeCoder(scheme=scheme)
        
        # Get available code names
        code_names = [c.name for c in codes]
        
        # Add segments manually (since we can't use hypothesis inside the test)
        original_segments = []
        for i in range(num_segments):
            doc_id = f"doc_{i}"
            text = f"测试文本片段{i}用于验证持久化"
            start_pos = 0
            end_pos = len(text)
            # Use first code for simplicity
            segment_codes = [code_names[i % len(code_names)]]
            note = f"备注{i}"
            
            segment = coder.add_segment(
                doc_id=doc_id,
                start=start_pos,
                end=end_pos,
                text=text,
                codes=segment_codes,
                note=note
            )
            if segment:
                original_segments.append({
                    'doc_id': doc_id,
                    'text': text,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'codes': segment_codes,
                    'note': note
                })
        
        assume(len(original_segments) >= 1)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, encoding='utf-8'
        ) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save the scheme and segments
            save_success = coder.save_scheme(tmp_path)
            assert save_success, "Save should succeed"
            
            # Create a new coder and load
            new_coder = QualitativeCoder()
            load_success = new_coder.load_scheme(tmp_path)
            assert load_success, "Load should succeed"
            
            # Verify segment count
            assert new_coder.get_segment_count() == len(original_segments), (
                f"Segment count mismatch: expected={len(original_segments)}, "
                f"actual={new_coder.get_segment_count()}"
            )
            
            # Verify each segment
            for i, orig_seg in enumerate(original_segments):
                loaded_seg = new_coder.get_segment(i)
                assert loaded_seg is not None, f"Segment {i} should exist"
                
                assert loaded_seg.document_id == orig_seg['doc_id'], (
                    f"Document ID mismatch for segment {i}"
                )
                assert loaded_seg.text == orig_seg['text'], (
                    f"Text mismatch for segment {i}"
                )
                assert loaded_seg.start_pos == orig_seg['start_pos'], (
                    f"Start position mismatch for segment {i}"
                )
                assert loaded_seg.end_pos == orig_seg['end_pos'], (
                    f"End position mismatch for segment {i}"
                )
                assert loaded_seg.codes == orig_seg['codes'], (
                    f"Codes mismatch for segment {i}: "
                    f"expected={orig_seg['codes']}, actual={loaded_seg.codes}"
                )
                assert loaded_seg.note == orig_seg['note'], (
                    f"Note mismatch for segment {i}"
                )
            
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @given(
        scheme_name=code_name_strategy,
        codes=codes_list_strategy(min_size=2, max_size=5)
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_hierarchical_codes_save_load_roundtrip(
        self, scheme_name: str, codes: list
    ):
        """
        **Feature: text-processing-enhancement, Property 1: 编码数据持久化一致性**
        **Validates: Requirements 1.6, 1.7**
        
        Property: For any coding scheme with hierarchical codes (parent-child),
        saving and loading should preserve the hierarchy structure.
        """
        assume(len(codes) >= 2)
        
        # Create a coding scheme
        scheme = CodingScheme(name=scheme_name)
        
        # Add first code as parent
        parent_code = codes[0]
        scheme.add_code(
            name=parent_code.name,
            description=parent_code.description,
            color=parent_code.color
        )
        
        # Add remaining codes as children of the first code
        child_names = []
        for code in codes[1:]:
            result = scheme.add_code(
                name=code.name,
                description=code.description,
                color=code.color,
                parent=parent_code.name
            )
            if result:
                child_names.append(code.name)
        
        assume(len(child_names) >= 1)
        
        # Create a coder with the scheme
        coder = QualitativeCoder(scheme=scheme)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, encoding='utf-8'
        ) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save the scheme
            save_success = coder.save_scheme(tmp_path)
            assert save_success, "Save should succeed"
            
            # Create a new coder and load
            new_coder = QualitativeCoder()
            load_success = new_coder.load_scheme(tmp_path)
            assert load_success, "Load should succeed"
            
            # Verify parent code exists
            loaded_parent = new_coder.scheme.get_code(parent_code.name)
            assert loaded_parent is not None, "Parent code should exist"
            assert loaded_parent.parent is None, "Parent code should have no parent"
            
            # Verify children
            loaded_children = new_coder.scheme.get_children(parent_code.name)
            assert set(loaded_children) == set(child_names), (
                f"Children mismatch: expected={set(child_names)}, "
                f"actual={set(loaded_children)}"
            )
            
            # Verify each child's parent reference
            for child_name in child_names:
                child_code = new_coder.scheme.get_code(child_name)
                assert child_code is not None, f"Child code '{child_name}' should exist"
                assert child_code.parent == parent_code.name, (
                    f"Child '{child_name}' should have parent '{parent_code.name}'"
                )
            
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @given(
        scheme_name=code_name_strategy,
        codes=codes_list_strategy(min_size=1, max_size=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_to_dict_from_dict_roundtrip(
        self, scheme_name: str, codes: list
    ):
        """
        **Feature: text-processing-enhancement, Property 1: 编码数据持久化一致性**
        **Validates: Requirements 1.6, 1.7**
        
        Property: For any QualitativeCoder, converting to dict and back
        should result in equivalent data.
        """
        assume(len(codes) >= 1)
        
        # Create a coding scheme with codes
        scheme = CodingScheme(name=scheme_name)
        for code in codes:
            scheme.add_code(
                name=code.name,
                description=code.description,
                color=code.color
            )
        
        # Create a coder with the scheme
        coder = QualitativeCoder(scheme=scheme)
        
        # Add some segments
        code_names = [c.name for c in codes]
        for i in range(3):
            coder.add_segment(
                doc_id=f"doc_{i}",
                start=0,
                end=10,
                text=f"文本片段{i}",
                codes=[code_names[i % len(code_names)]],
                note=f"备注{i}"
            )
        
        # Convert to dict
        data_dict = coder.to_dict()
        
        # Create new coder from dict
        new_coder = QualitativeCoder.from_dict(data_dict)
        
        # Verify scheme
        assert new_coder.scheme.name == scheme_name
        assert new_coder.scheme.get_code_count() == len(codes)
        
        # Verify codes
        for code in codes:
            loaded_code = new_coder.scheme.get_code(code.name)
            assert loaded_code is not None
            assert loaded_code.description == code.description
            assert loaded_code.color == code.color
        
        # Verify segments
        assert new_coder.get_segment_count() == coder.get_segment_count()
        
        for i in range(coder.get_segment_count()):
            orig_seg = coder.get_segment(i)
            loaded_seg = new_coder.get_segment(i)
            
            assert loaded_seg.document_id == orig_seg.document_id
            assert loaded_seg.text == orig_seg.text
            assert loaded_seg.codes == orig_seg.codes

    @given(codes=codes_list_strategy(min_size=1, max_size=5))
    @settings(max_examples=100, deadline=None)
    def test_empty_segments_save_load(self, codes: list):
        """
        **Feature: text-processing-enhancement, Property 1: 编码数据持久化一致性**
        **Validates: Requirements 1.6, 1.7**
        
        Property: A coding scheme with codes but no segments should
        save and load correctly.
        """
        assume(len(codes) >= 1)
        
        # Create a coding scheme with codes but no segments
        scheme = CodingScheme(name="test_scheme")
        for code in codes:
            scheme.add_code(
                name=code.name,
                description=code.description,
                color=code.color
            )
        
        coder = QualitativeCoder(scheme=scheme)
        
        # Verify no segments
        assert coder.get_segment_count() == 0
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, encoding='utf-8'
        ) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save
            save_success = coder.save_scheme(tmp_path)
            assert save_success, "Save should succeed"
            
            # Load
            new_coder = QualitativeCoder()
            load_success = new_coder.load_scheme(tmp_path)
            assert load_success, "Load should succeed"
            
            # Verify codes are preserved
            assert new_coder.scheme.get_code_count() == len(codes)
            
            # Verify no segments
            assert new_coder.get_segment_count() == 0
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
