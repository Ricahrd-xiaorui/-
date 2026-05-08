# -*- coding: utf-8 -*-
"""
数据导出工具 - 统一管理CSV导出功能
"""

import pandas as pd
from io import StringIO


def dataframe_to_csv(df: pd.DataFrame, index: bool = False) -> str:
    """
    将DataFrame转换为CSV字符串

    Args:
        df: DataFrame对象
        index: 是否包含索引列

    Returns:
        str: CSV格式字符串（UTF-8 BOM编码）
    """
    return df.to_csv(index=index, encoding='utf-8-sig')


def dict_to_csv(data: list, columns: list = None) -> str:
    """
    将字典列表转换为CSV字符串

    Args:
        data: 字典列表
        columns: 列名列表，如果为None则使用字典的键

    Returns:
        str: CSV格式字符串
    """
    if not data:
        return ""
    df = pd.DataFrame(data, columns=columns)
    return dataframe_to_csv(df)


def download_csv_button(data: str, filename: str, label: str = "下载 CSV") -> None:
    """
    创建CSV下载按钮的辅助函数

    注意：此函数需要在Streamlit上下文中调用

    Args:
        data: CSV字符串
        filename: 下载文件名
        label: 按钮标签
    """
    import streamlit as st
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime="text/csv"
    )


def export_key_value_data(data: dict, key_name: str = "名称", value_name: str = "值") -> str:
    """
    将键值对字典导出为CSV

    Args:
        data: 键值对字典
        key_name: 键列名
        value_name: 值列名

    Returns:
        str: CSV格式字符串
    """
    df = pd.DataFrame(list(data.items()), columns=[key_name, value_name])
    return dataframe_to_csv(df)


def export_two_column_data(data: list, col1_name: str, col2_name: str) -> str:
    """
    将两列元组列表导出为CSV

    Args:
        data: [(value1, value2), ...] 格式的列表
        col1_name: 第一列列名
        col2_name: 第二列列名

    Returns:
        str: CSV格式字符串
    """
    df = pd.DataFrame(data, columns=[col1_name, col2_name])
    return dataframe_to_csv(df)


def export_frequency_data(word_freq: dict, word_col: str = "词语", freq_col: str = "频率") -> str:
    """
    将词频数据导出为CSV

    Args:
        word_freq: 词频字典 {词语: 频率}
        word_col: 词语列名
        freq_col: 频率列名

    Returns:
        str: CSV格式字符串
    """
    sorted_data = sorted(word_freq.items(), key=lambda x: -x[1])
    df = pd.DataFrame(sorted_data, columns=[word_col, freq_col])
    return dataframe_to_csv(df)


def export_cooccurrence_data(cooccurrence: dict,
                             word1_col: str = "词语1",
                             word2_col: str = "词语2",
                             freq_col: str = "共现频率") -> str:
    """
    将共现数据导出为CSV

    Args:
        cooccurrence: 共现字典 {(词语1, 词语2): 频率}
        word1_col: 第一词列名
        word2_col: 第二词列名
        freq_col: 频率列名

    Returns:
        str: CSV格式字符串
    """
    sorted_data = sorted(cooccurrence.items(), key=lambda x: -x[1])
    rows = [(pair[0], pair[1], freq) for pair, freq in sorted_data]
    df = pd.DataFrame(rows, columns=[word1_col, word2_col, freq_col])
    return dataframe_to_csv(df)


def export_network_nodes(nodes: list) -> str:
    """
    导出网络节点数据为CSV

    Args:
        nodes: 节点列表，每个节点是包含id, label, size等字段的字典

    Returns:
        str: CSV格式字符串
    """
    if not nodes:
        return ""
    df = pd.DataFrame(nodes)
    return dataframe_to_csv(df)


def export_network_edges(edges: list) -> str:
    """
    导出网络边数据为CSV

    Args:
        edges: 边列表，每个边是包含source, target, weight字段的字典

    Returns:
        str: CSV格式字符串
    """
    if not edges:
        return ""
    df = pd.DataFrame(edges)
    return dataframe_to_csv(df)
