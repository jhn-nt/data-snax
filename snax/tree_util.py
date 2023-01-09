import jax.numpy as jnp
from jax.tree_util import *
from jax import jit

from jaxtyping import PyTree, Array
from typing import Optional, Union


@jit
def tree_stack(a: PyTree, b: PyTree) -> PyTree:
    """Stacks Pytrees leaves.

    'a' and 'b' must have the same 'PyTreeDef' otherwise an error is raised.

    Parameters
    ----------
    a : PyTree
        Input a.
    b : PyTree
        Input b.

    Returns
    -------
    PyTree
        A PyTree with leaves stacked from 'a'and 'b'.
    """
    a_data, a_def = tree_flatten(a)
    b_data, b_def = tree_flatten(b)

    assert (
        a_def == b_def
    ), f"'a' and 'b' must have the same 'PyTreeDef', found: {a_def} and {b_def}"

    c = []
    for c_a, c_b in zip(a_data, b_data):
        c.append(jnp.vstack((c_a, c_b)))
    return tree_unflatten(a_def, c)


def tree_height(tree: PyTree, val: Optional[int] = None) -> int:
    """Returnsthe 'height' of a tree.

    The 'height' of a tree is simply the size of first dimension of each leaves.

    Parameters
    ----------
    tree : PyTree
        Input.
    val : Optional[int], optional
        index of the leaf to which to measure the height.
        If None, it is assumed that all leaves have same height and returns thei height, by default None

    Returns
    -------
    int
        Height of tree.
    """
    heights = tree_leaves(tree_map(lambda x: jnp.atleast_2d(x).shape[0], tree))
    if val is None:
        assert len(set(heights)) == 1
        val = 0

    return heights[val]


def tree_index(tree: PyTree, ix: Union[int, Array]) -> PyTree:
    """Indexes the leaves of a tree.

    Parameters
    ----------
    tree : PyTree
        Input.
    ix : Union[int, Array]
        index or indexes.

    Returns
    -------
    PyTree
        Indexed elem of a tree.
    """
    return tree_map(lambda x: jnp.atleast_2d(x)[jnp.array(ix)], tree)


def tree_update(base_tree: PyTree, ix: Array, updates: PyTree) -> PyTree:
    """Update a tree.

    Update 'base_tree' in each leaf at 'ix' with 'updates'.

    Parameters
    ----------
    base_tree : PyTree
        Input.
    ix : Array
        Index or Indexes.
    updates : PyTree
        Tree with updates.

    Returns
    -------
    PyTree
        Updated pytree.
    """
    assert tree_structure(base_tree) == tree_structure(updates)

    data, tree_def = tree_flatten(base_tree)
    updates = tree_leaves(updates)

    updated_data = []
    for (c_d, c_u) in zip(data, updates):
        updated_data.append(c_d.at[jnp.array(ix)].set(c_u))

    return tree_unflatten(tree_def, updated_data)
