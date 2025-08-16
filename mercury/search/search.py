# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

import copy
import itertools
from mercury.ir.primitives import parallelize, shift
from mercury.ir.distributed import DeviceMesh
from mercury.ir.init_distributed import init_distributed
from mercury.ir.nodes import Program
from mercury.ir.elements import Axis
from mercury.ir.tile import tile_loop
from mercury.ir.utils import collect_axis, collect_loops, collect_parallelizeable_axes, get_buffers
from typing import List, Iterator, Tuple


def enumerate_mesh_shapes(mesh_size: int, max_dim: int, current_shape: List[int] = None, remaining: int = None) -> Iterator[Tuple[int, ...]]:
    """
    enumerate all possible device mesh shapes.
    Parameters
    ----------
    mesh_size : int
        The size of the original 1D mesh
    max_dim : int
        The maximum number of dimensions allowed
    current_shape : List[int], optional
        The current shape being constructed, used for recursion
    remaining : int, optional
        The remaining size to be allocated, used for recursion
    Yields
    -------
    Tuple[int, ...]
        A possible mesh shape, e.g., (4,), (2,2), (2,2,2), etc.
    """
    if current_shape is None:
        current_shape = []
        remaining = mesh_size
    
    # base case: if the current shape has reached max_dim but there is still remaining size, return
    if len(current_shape) >= max_dim and remaining > 1:
        return
        
    # if no remaining size to allocate, yield the current shape
    if remaining == 1:
        yield tuple(current_shape)
        return
    elif remaining < 1:
        return
        
    # try all possible factors
    for factor in range(2, remaining + 1):
        if remaining % factor == 0:
            # for each factor, recursively enumerate the remaining size
            for shape in enumerate_mesh_shapes(mesh_size, max_dim, 
                                            current_shape + [factor], 
                                            remaining // factor):
                yield shape

def enumerate_mesh_assignment(ndim: int, axes_num: int) -> Iterator[List[Tuple[int, int]]]:
    """
    enumerate all possible mesh dimension assignment for a given number of axes and mesh dimensions.
    each axis must be assigned to a continuous range of dimensions, and all dimensions must be assigned.
    For example, for a 2D mesh and 2 axes, the possible assignments include:
    - axis1: [0,1], axis2: []
    - axis1: [], axis2: [0,1]
    - axis1: [0], axis2: [1]
    - axis1: [1], axis2: [0]
    Parameters
    ----------
    ndim : int
        The total number of dimensions in the mesh
    axes_num : int
        The number of axes to be assigned
    Yields
    -------
    List[Tuple[int, int]]
        A possible assignment of axes to dimensions, where each tuple represents (start_dim, num_dims)
    """
    def _recursive_assign(remaining_dims: List[int], axis_idx: int, 
                         current_assignment: List[Tuple[int, int]]) -> Iterator[List[Tuple[int, int]]]:
        # base case: if all axes have been assigned
        if axis_idx == axes_num:
            # only if all dimensions are assigned, it is a valid assignment
            if not remaining_dims:
                yield copy.deepcopy(current_assignment)
            return

        # try to assign no dimensions to the current axis
        yield from _recursive_assign(remaining_dims, axis_idx + 1, 
                                   current_assignment + [(0, 0)])

        # try to assign a continuous range of dimensions to the current axis
        for start in range(len(remaining_dims)):
            for length in range(1, len(remaining_dims) - start + 1):
                # get the dimensions to be assigned
                assigned_dims = remaining_dims[start:start + length]
                # check if the assigned dimensions are continuous
                if assigned_dims != list(range(assigned_dims[0], assigned_dims[-1] + 1)):
                    continue

                remaining = remaining_dims[:start] + remaining_dims[start + length:]
                
                yield from _recursive_assign(remaining, axis_idx + 1,
                                          current_assignment + [(assigned_dims[0], length)])

    yield from _recursive_assign(list(range(ndim)), 0, [])

def enumerate_axis_split(axes: List[Axis], res_cards: int, cur_split: List[int]) -> Iterator[List[int]]:
    """ enumarate all possible axis split for a given axis number and resource cards """
    if len(cur_split) == len(axes):
        yield copy.deepcopy(cur_split)
        return
    
    cur_axis = axes[len(cur_split)]
    
    for i in range(1, res_cards + 1):
        if cur_axis.size % i == 0 and cur_axis.size // i >= cur_axis.min_block_size \
        and res_cards % i == 0:
            yield from enumerate_axis_split(axes, res_cards // i, cur_split + [i])


# def detect_conflict_axis(assign, axes_list, mutex_pair) -> bool:
#     axes = [axis.name for axes in axes_list for axis in axes]
#     non_zero_axes = []
#     for axis, (start, length) in zip(axes, assign):
#         if length > 0:
#             non_zero_axes.append(axis)
            
#     for axis in non_zero_axes:
#         if mutex_pair.get(axis) is None:
#             continue
#         for banned in mutex_pair[axis]:
#             if banned in non_zero_axes:
#                 return True
#     return False

def search(input_program: Program, origin_mesh: DeviceMesh, split_axis_names: List[str] = list()) -> Iterator[Program]:
    """
    Search for the best schedule for a given program.

    Parameters
    ----------
    program : Program
        The program to search for the best schedule.

    mesh : DeviceMesh
        The device mesh to search for the best schedule.
        we assume it is a 1D mesh for now
        TODO: support 2D mesh

    split_axis_names : List[str]
        The axes can be split to multiple dimensions.

    Returns
    -------
    Iterator[Program]
        An iterator of programs with different schedules.
    """

    axes_list = input_program.visit(collect_parallelizeable_axes) # list of list of axes, each list coresponds to a GridLoop
    # print(f"axes_list: {axes_list}")
    axis_num = sum(len(axes) for axes in axes_list)
    # print(f"axis_num: {axis_num}")
    
    max_dim = axis_num + len(split_axis_names) # allow one axis to be assigned to two dimensions, for double ring

    # the following code is found to be wrong, q and kv can be parallelized together
    # # find the axis pairs that can't be parallelized together
    # # theortically, we can analyze the data dependency to get the pairs
    # # here we will simplify the condition to be if match the
    # # pattern of q[axis0, axis1], k[axis0, axis2] then axis1 and axis2 can't be parallelized together
    # mutex_pair = {}

    # buffers = input_program.visit(get_buffers)
    # for buffer1 in buffers:
    #     for buffer2 in buffers:
    #         if buffer1 == buffer2:
    #             continue
    #         axes1 = []
    #         for axes in buffer1.bound_axes:
    #             for axis in axes:
    #                 axes1.append(axis.name)
    #         axes2 = []
    #         for axes in buffer2.bound_axes:
    #             for axis in axes:
    #                 axes2.append(axis.name)
            
    #         # compute common axes
    #         common_axes = set(axes1) & set(axes2)
    #         if len(common_axes) < len(axes1) and len(common_axes) < len(axes2):
    #             axes1 = list(set(axes1) - common_axes)
    #             axes2 = list(set(axes2) - common_axes)
    #             for axis in axes1:
    #                 if mutex_pair.get(axis) is None:
    #                     mutex_pair[axis] = set()
    #                 mutex_pair[axis].update(axes2)
    #             for axis in axes2:
    #                 if mutex_pair.get(axis) is None:
    #                     mutex_pair[axis] = set()
    #                 mutex_pair[axis].update(axes1)

    axes_split = []
    for axes in axes_list:
        for axis in axes:
            if axis.name in split_axis_names:
                axes_split.append(axis)

    assert len(axes_split) == len(split_axis_names), "split axis not found in the program"

    for split_num in enumerate_axis_split(axes_split, len(origin_mesh.devices), []):
        # try all possible axis split

        splited_program = copy.deepcopy(input_program)

        # do the split
        for axis_name, split in zip(split_axis_names, split_num):
            if split == 1:
                continue
            axes = splited_program.visit(collect_axis)
            target_axis = None
            for axis in axes:
                if axis.name == axis_name:
                    target_axis = axis
                    break
            assert target_axis is not None, "split axis not found in the program"
            tile_loop(splited_program, target_axis, target_axis.size // split)

        axes = splited_program.visit(collect_axis)
        axis_num = len(axes)

        for mesh_shape in enumerate_mesh_shapes(len(origin_mesh.devices), max_dim):
            # try all possible mesh shapes
            
            program_shape = copy.deepcopy(splited_program)
            mesh = origin_mesh.reshape(mesh_shape)

            init_distributed(program_shape, mesh)

            # we need to partition the mesh for each axis


            for assign in enumerate_mesh_assignment(len(mesh_shape), axis_num):
                # try all possible assignments
                program = copy.deepcopy(program_shape)

                seed_program = copy.deepcopy(program)

                loops = program.visit(collect_loops)
                axes_list = program.visit(collect_parallelizeable_axes)

                all_axes = program.visit(collect_axis)
                axes_names = set([axis.name for axis in all_axes])
                ringable_axes = set()

                # collect the axes to be parallelized
                parallelize_axes = []
                cnt = 0
                for loop, axes in zip(loops, axes_list):
                    for axis in axes:
                        if assign[cnt][1] != 0:
                            parallelize_axes.append(axis.name)
                        cnt += 1

                cnt = 0
                succ = True
                for loop, axes in zip(loops, axes_list):
                    for axis in axes:
                        usd_axes = set(parallelize_axes)
                        succ &= parallelize(program, loop, axis, mesh, assign[cnt][0], assign[cnt][0] + assign[cnt][1], usd_axes)
                        if not succ:
                            break
                        if assign[cnt][1] != 0:
                            ringable_axes |= axes_names - usd_axes
                        shift(program, axis, mesh, assign[cnt][0], assign[cnt][0] + assign[cnt][1], 1, usd_axes)
                        cnt += 1
                    if not succ:
                        break
                
                if succ:
                    # print(f"mesh_shape: {mesh_shape}")
                    # print(f"assign: {assign}")
                    yield program

                    # enumerate all subset of ringable axes
                    for i in range(1, len(ringable_axes) + 1): # start from 1, as empty set is the same as above
                        for ringable_axes_subset in itertools.combinations(ringable_axes, i):
                            program = copy.deepcopy(seed_program)
                            loops = program.visit(collect_loops)
                            axes_list = program.visit(collect_parallelizeable_axes)
                            cnt = 0
                            for loop, axes in zip(loops, axes_list):
                                for axis in axes:
                                    usd_axes = set(ringable_axes_subset)
                                    usd_axes.update(parallelize_axes)
                                    succ = parallelize(program, loop, axis, mesh, assign[cnt][0], assign[cnt][0] + assign[cnt][1], usd_axes)
                                    assert succ == True, "should be able to parallelize as we have checked before"
                                    shift(program, axis, mesh, assign[cnt][0], assign[cnt][0] + assign[cnt][1], 1, usd_axes)
                                    cnt += 1
                            yield program