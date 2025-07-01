"""
Tests for IR visitor pattern implementation.
"""
import pytest
from typing import Optional, List, cast
from mercury.ir.nodes import (
    Program, AxisDef, BufferMatch, GridLoop,
    BufferStore, BufferLoad, IRNode
)
from mercury.ir.elements import Axis, Buffer

def test_basic_visitor():
    """Test basic visitor functionality on simple nodes."""
    # Create test nodes
    axis = Axis("I", 10)
    axis_def = AxisDef(axis)
    
    # Test simple visitor that counts nodes
    def count_nodes(node: IRNode) -> int:
        return 1
        
    assert len(axis_def.visit(count_nodes)) == 1

def test_nested_visitor():
    """Test visitor on nested IR structure."""
    axis = Axis("I", 10)
    buffer = Buffer(None, (10,), [[axis]], [[1]])
    
    program = Program(
        name="test",
        inputs=[],
        defaults=[],
        outputs=[],
        body=[
            AxisDef(axis),
            BufferMatch(buffer, "A"),
            BufferStore(buffer, [0], 1.0)
        ]
    )
    
    # Count each type of node
    counts = {}
    def count_by_type(node: IRNode) -> None:
        type_name = type(node).__name__
        counts[type_name] = counts.get(type_name, 0) + 1
        return None
    
    program.visit(count_by_type)
    assert counts["Program"] == 1
    assert counts["AxisDef"] == 1
    assert counts["BufferMatch"] == 1
    assert counts["BufferStore"] == 1

def test_conditional_visitor():
    """Test visitor with conditional collection."""
    axis_i = Axis("I", 10)
    axis_j = Axis("J", 20)
    buffer = Buffer(None, (10, 20), [[axis_i], [axis_j]], [[1], [1]])
    
    program = Program(
        name="test",
        inputs=[],
        outputs=[],
        defaults=[],
        body=[
            AxisDef(axis_i),
            AxisDef(axis_j),
            GridLoop(
                axes=[axis_i, axis_j],
                axis_types="ss",
                body=[
                    BufferStore(buffer, [0, 0], 1.0)
                ]
            )
        ]
    )
    
    # Find all nodes using axis_i
    def uses_axis_i(node: IRNode) -> Optional[IRNode]:
        if isinstance(node, (AxisDef, GridLoop)):
            if isinstance(node, AxisDef) and node.axis == axis_i:
                return node
            elif isinstance(node, GridLoop) and axis_i in node.axes:
                return node
        return None
    
    using_i = program.visit(uses_axis_i)
    assert len(using_i) == 2  # AxisDef and GridLoop

def test_visitor_return_types():
    """Test visitor with different return types."""
    axis = Axis("I", 10)
    buffer = Buffer(None, (10,), [[axis]], [[1]])
    store = BufferStore(buffer, [0], 1.0)
    
    # Return integers
    def count_depth(node: IRNode) -> int:
        return 1
    depths = store.visit(count_depth)
    assert all(isinstance(d, int) for d in depths)
    
    # Return strings
    def node_names(node: IRNode) -> str:
        return type(node).__name__
    names = store.visit(node_names)
    assert all(isinstance(n, str) for n in names)
    
    # Return None for filtering
    def only_stores(node: IRNode) -> Optional[BufferStore]:
        return node if isinstance(node, BufferStore) else None
    stores = store.visit(only_stores)
    assert all(isinstance(s, BufferStore) for s in stores)

if __name__ == "__main__":
    pytest.main([__file__])