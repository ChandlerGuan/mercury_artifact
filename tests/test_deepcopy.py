import unittest
import copy
from mercury.ir.nodes import IRNode, Program, GridLoop, RingComm, PyNode, BufferStore, BufferLoad, AxisDef
from mercury.ir.elements import Axis, Buffer
from mercury.ir.distributed import DeviceMesh, ShardingSpec, ShardType
import ast

def collect_node_types(node: IRNode) -> list:
    def visitor(n: IRNode) -> tuple:
        return n
        
    return node.visit(visitor)

class TestDeepCopy(unittest.TestCase):
    def test_basic_copy(self):
        axis = Axis("i", 128)
        comm = RingComm(axis=axis, num_cards=4, name="comm1", shard_dim=1)
        
        copied = copy.deepcopy(comm)
        
        self.assertEqual(copied.name, comm.name)
        self.assertEqual(copied.num_cards, comm.num_cards)
        self.assertEqual(copied.axis.name, comm.axis.name)
        self.assertEqual(copied.axis.size, comm.axis.size)
        
        self.assertIsNot(copied, comm)
        self.assertIsNot(copied.axis, comm.axis)

    def test_shared_nodes(self):
        shared_axis = Axis("k", 64)
        
        comm1 = RingComm(shard_dim=1, axis=shared_axis, num_cards=4, name="comm1")
        comm2 = RingComm(shard_dim=1, axis=shared_axis, num_cards=4, name="comm2")
        
        prog = Program(
            name="test",
            inputs=[],
            defaults=[],
            outputs=None,
            body=[comm1, comm2]
        )
        
        copied = copy.deepcopy(prog)
        
        copied_comm1 = copied.body[0]
        copied_comm2 = copied.body[1]
        self.assertIs(copied_comm1.axis, copied_comm2.axis)
        
        self.assertIsNot(copied_comm1.axis, comm1.axis)

    def test_complex_structure(self):
        mesh = DeviceMesh(devices=[0,1,2,3], shape=(2,2))
        
        axis = Axis("i", 128)
        buffer = Buffer(
            tensor="X",
            shape=[128, 128],
            bound_axes=[[axis], []],
            axes_factor=[[1], [1]],
            shard_spec=ShardingSpec(
                mesh=mesh,
                specs=[(ShardType.SHARD, [0]), ShardType.REPLICATE]
            )
        )
        
        node = PyNode(node=ast.Name(id='x', ctx=ast.Load()))
        grid = GridLoop(
            axes=[axis],
            axis_types="s",
            body=[
                BufferStore(buffer=buffer, indices=[axis, 0], value=node),
                BufferLoad(buffer=buffer, indices=[axis, 0], target="y")
            ]
        )
        
        prog = Program(
            name="complex",
            inputs=[],
            defaults=[],
            outputs=None,
            body=[grid],
            mesh=mesh
        )
        
        copied = copy.deepcopy(prog)
        
        self.assertEqual(len(copied.body), 1)
        copied_grid = copied.body[0]
        self.assertEqual(len(copied_grid.body), 2)
        
        self.assertEqual(copied.mesh.shape, prog.mesh.shape)
        self.assertIsNot(copied.mesh, prog.mesh)
        
        copied_store = copied_grid.body[0]
        self.assertEqual(copied_store.buffer.shard_spec.specs[0][0], ShardType.SHARD)
        self.assertEqual(copied_store.buffer.shard_spec.specs[1], ShardType.REPLICATE)

    def test_independence(self):
        axis = Axis("i", 128)
        buffer = Buffer("X", [128], bound_axes=[[axis]], axes_factor=[[1]])
        store = BufferStore(buffer=buffer, indices=[axis], value="1.0")
        
        copied = copy.deepcopy(store)
        
        axis.size = 256
        buffer.tensor = "Y"
        
        self.assertEqual(copied.buffer.bound_axes[0][0].size, 128)
        self.assertEqual(copied.buffer.tensor, "X")
        self.assertEqual(copied.indices[0].size, 128)
        
    def test_transformed_ir_copy(self):
        import ast
        import textwrap
        from mercury.frontend.parser import IRBuilder
        from mercury.ir.distributed import DeviceMesh
        from mercury.ir.init_distributed import init_distributed
        import mercury.ir.primitives as sp
        import mercury.ir.loop_eliminating as le
        from utils.flash_attn_dsl import flash_attn_pack_kv_double_ring_template
        
        batch_size, seqlen, nheads, dim = 4, 4096, 5, 128
        source = flash_attn_pack_kv_double_ring_template.format(
            BATCH=batch_size,
            SEQ_LEN=seqlen,
            HEADS=nheads,
            HEAD_DIM=dim,
            SEQ_LEN_IN=seqlen // 2,
            SEQ_LEN_OUT=2,
        )
        
        tree = ast.parse(textwrap.dedent(source))
        builder = IRBuilder()
        program = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                program = builder.visit(node)
                break
                
        self.assertIsNotNone(program, "Failed to parse IR")
        
        world_size = 8
        devices = [i for i in range(world_size)]
        mesh = DeviceMesh(devices, (world_size // 2, 2))
        
        init_distributed(program, mesh)
        
        axes = [n for n in program.visit(lambda n: n if isinstance(n, AxisDef) else None) if n is not None]
        query_axis = axes[2]  # S_q axis
        
        loops = [node for node in program.visit(lambda n: n if isinstance(n, GridLoop) else None)]
        outer_loop = next(l for l in loops if len(l.axes) == 3)
        
        sp.parallelize(program, outer_loop, query_axis.axis, mesh, 0, 2)
        sp.shift(program, query_axis.axis, mesh, 0, 2, 1)
        le.eliminate_loops(program)
        
        copied = copy.deepcopy(program)
        self.assertIsNot(copied, program)
        
        original_structure = collect_node_types(program)
        copied_structure = collect_node_types(copied)
        self.assertEqual(original_structure, copied_structure)
        for origin, copied in zip(original_structure, copied_structure):
            self.assertIsNot(origin, copied)

if __name__ == '__main__':
    unittest.main()