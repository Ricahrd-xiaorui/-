# 语义网络可视化修复总结

## 问题描述
用户报告语义网络模块显示统计信息（9077个节点，55321条边），但没有显示可视化图表，提示"没有可显示的节点（当前显示 0 个）"。

## 根本原因
之前的修复中，`_render_network_visualization()` 函数被不完整地编辑，导致：
1. 代码中存在重复的渲染逻辑
2. 存在引用未定义变量的 `else` 分支（`edges`, `pos`, `node_x` 等）
3. 存在孤立的 `except` 块，破坏了代码结构
4. 虽然语法上可以通过某些检查，但运行时会出现问题

## 修复内容

### 1. 清理 `_render_network_visualization()` 函数 (lines 570-720)
- 移除了重复的可视化渲染代码
- 移除了引用未定义变量的 `else` 分支
- 移除了孤立的 `except ImportError` 块
- 保留了正确的按钮触发逻辑：
  - 点击"生成网络图"按钮时才执行可视化
  - 未点击时显示提示信息

### 2. 添加调试信息
为了帮助诊断问题，添加了详细的调试输出：

**在 `_render_network_visualization()` 中：**
```python
# 调试信息
st.info(f"🔍 调试：网络有 {num_nodes} 个节点，准备提取前 {max_nodes} 个")

# 获取可视化数据
nodes, edges = builder.to_vis_data(max_nodes)

st.info(f"🔍 调试：to_vis_data 返回了 {len(nodes)} 个节点，{len(edges)} 条边")

if not nodes:
    st.warning("没有可显示的节点")
    st.error("⚠️ 这可能是一个bug，请检查网络数据")
    # 显示网络的基本信息用于调试
    if builder.network:
        sample_nodes = list(builder.network.nodes())[:5]
        st.write(f"网络中的前5个节点示例: {sample_nodes}")
    return
```

**在 `render_semantic_network()` 中：**
```python
# 调试信息
st.write(f"🔍 调试：从session_state读取网络，节点数={network.number_of_nodes() if network else 0}")

if builder is None:
    st.write("🔍 调试：builder为None，正在重建...")
    builder = SemanticNetworkBuilder(texts, cooccurrence_data)
    builder.network = network
    builder._community_labels = communities
    builder._centrality_metrics = centrality
    st.write(f"🔍 调试：重建后 builder.network 节点数={builder.network.number_of_nodes() if builder.network else 0}")
else:
    st.write(f"🔍 调试：使用已有builder，节点数={builder.network.number_of_nodes() if builder.network else 0}")
```

## 修复后的工作流程

1. **网络构建**：用户点击"构建语义网络"
   - 对于大网络（>50节点），不自动计算社区和中心性
   - 网络对象保存到 `st.session_state["semantic_network"]`
   - Builder对象保存到 `st.session_state["semantic_network_builder"]`

2. **显示统计信息**：显示节点数、边数等基本统计

3. **可视化**：用户切换到"网络可视化"标签
   - 显示网络大小信息和警告（如果网络较大）
   - 显示可视化设置（布局算法、颜色依据、标签显示）
   - 显示"生成网络图"按钮
   - 点击按钮后：
     - 调用 `builder.to_vis_data(max_nodes)` 获取前N个节点
     - 计算布局
     - 渲染Plotly图表

4. **调试输出**：在每个关键步骤显示调试信息，帮助定位问题

## 预期结果

修复后，用户应该能够：
1. 看到详细的调试信息，了解数据流转情况
2. 点击"生成网络图"按钮后看到可视化图表
3. 如果仍然没有节点，调试信息会显示具体原因

## 下一步

如果用户仍然看不到可视化：
1. 检查调试输出，确认：
   - 网络对象是否正确保存和恢复
   - `to_vis_data()` 是否返回了节点数据
   - 是否有任何错误信息
2. 可能需要检查：
   - `cooccurrence_data` 是否正确
   - 网络构建时的 `min_weight` 参数是否过高
   - 是否有其他运行时错误

## 文件修改

- `modules/semantic_network.py`
  - 修复 `_render_network_visualization()` 函数（lines 570-720）
  - 添加调试信息到 `render_semantic_network()` 函数（lines 535-545）
  - 添加调试信息到 `_render_network_visualization()` 函数（lines 605-620）
