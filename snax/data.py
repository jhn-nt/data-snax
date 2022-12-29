import jax.numpy as jnp
from jax.tree_util import *
from jax import vmap, jit
from jax.random import PRNGKey, uniform, permutation
from jaxlib.xla_extension.pytree import PyTreeDef
from jaxlib.xla_extension import DeviceArray, CompiledFunction

import numpy as np
from datetime import datetime
from typing import Any, Tuple, Union, Callable, Type, Iterable
from numpy.typing import NDArray
import time


# primitives
def leaves_sizes(tree: Any) -> Tuple[int]:
    """Return the size of the first dimension of each leaf in tree.



    Parameters
    ----------
    tree : Any
        Any tree.

    Returns
    -------
    Tuple[int]
        A tuple of length equals to the number of leaves where each element reprents the size of the first dimension.
    """
    data = tree_leaves(tree)
    return tuple(map(lambda x: x.shape[0], data))


def index_elems_in_tree(indices: Union[range, int, NDArray], tree: Any) -> Any:
    """Indexes pytrees based on indices or slices.

    Parameters
    ----------
    indices : Union[range,int,NDArray]
        Indices of the slices.
    tree : Any
        Input tree.

    Returns
    -------
    Any
        The sliced tree according to the indices.
    """
    data = tree_leaves(tree)
    tree_def = tree_structure(tree)
    sliced_data = tuple(map(lambda x: x.take(indices, axis=0), data))
    return build_tree(tree_def, sliced_data)


def to_jax_pytree(tree: Any) -> Any:
    """Converts a numpy array based pytree to a jax equivalent.

    Parameters
    ----------
    tree : Any
        numpy-based pytree.

    Returns
    -------
    Any
        jax-based pytree.
    """
    data, tree_def = tree_flatten(tree)
    return tree_unflatten(tree_def, list(map(jnp.asarray, data)))


@jit
def nested_indexer(
    ix: Union[NDArray, int], inner_array: NDArray, outer_array: NDArray
) -> NDArray:
    return outer_array[inner_array[ix]]


def get_batch_schedule(batch_size:int,indices:NDArray,cardinality:int)->Tuple[NDArray,int]:
    """Generates a schedule of batches pointers based on `batch_size`. 

    Parameters
    ----------
    batch_size : int
        Desired batch size.
    indices : NDArray
        Observations indexes.
    cardinality : int
        Number of observations.

    Returns
    -------
    Tuple[NDArray,int]
        batch_indexes: matrix where each ithrow represent the indexes of the ith batch.
    """
    n_batches = cardinality // batch_size
    batch_indexes = jnp.reshape(
        indices[: n_batches * batch_size], (n_batches, batch_size)
    )
    return batch_indexes, n_batches

# utils
def update_feed(
    cardinality: int, reader: Callable, tree_def: PyTreeDef, indices: Callable
) -> dict:
    """Generates a dictionary with keys compatible to a `snax.data.Dataset` object.

    This methods serves also to assert correctness of types.

    Parameters
    ----------
    cardinality : int
    reader : Callable
    tree_def : PyTreeDef
    indices : Callable

    Returns
    -------
    dict
    """
    assert isinstance(
        cardinality, int
    ), f"`cardinality` must be of type `int`, found: {type(cardinality)}"
    assert isinstance(
        reader, Callable
    ), f"`reader` must be of type `Callable`, found: {type(reader)}"
    assert isinstance(
        tree_def, PyTreeDef
    ), f"`tree_def` must be of type `PyTreeDef`, found: {type(tree_def)}"
    assert isinstance(
        indices, Callable
    ), f"`indices` must be of type `Callable`, found: {type(indices)}"

    return {
        "cardinality": cardinality,
        "reader": reader,
        "tree_def": tree_def,
        "indices": indices,
    }


class Dataset:
    """A jax interpretation of tensorflow `tf.data.Dataset`.

    In this current verions it supports:
    #. `map`: behavior equal to that of `tf.data.Dataset.map`
    #. `take`: behavior equal to that of `tf.data.Dataset.take`
    #. `skip`: behavior equal to that of `tf.data.Dataset.skip`
    #. `zip`: behavior equal to that of `tf.data.Dataset.zip`
    #. `shuffle`: behavior is equal to `tf.data.Dataset.shuffle` but with different input signature,
        instead of a `shuffle_buffer` it requires a `jax.PRNGKEY`.
    #. `batch`: similar behavior to `tf.data.Dataset.batch` but completely different mechanics.
        No other transformation can be applied after a `batch` call.

    What it does not yet support:
    #. `apply`
    #. `from_generator`
    #. `concatenate`
    #. `bucket_by_sequence_lenght`
    #. `filter`
    
    """
    def __init__(self, feeds: dict):
        self.feeds = feeds
        self.epoch=0

    def map(self, func: Callable) -> "Dataset":
        """Maps `func` to the data.

        `func` is applied to each observation independently.

        Parameters
        ----------
        func : Callable
            Desired transofrmation to apply to the dataset.

        Returns
        -------
        Dataset
            Mapped dataset.
        """
        def mapped_reader(indices):
            return vmap(func)(self.feeds["reader"](indices))

        return Dataset(
            {
                "cardinality": self.feeds["cardinality"],
                "reader": mapped_reader,
                "tree_def": tree_structure(mapped_reader(jnp.arange(2))),
                "indices": self.feeds["indices"],
            }
        )

    def take(self, elems: int) -> "Dataset":
        """Takes the first `elems` of the dataset.

        Parameters
        ----------
        elems : int
            desired number of observations to consider.

        Returns
        -------
        Dataset
            Taken Dataset.
        """
        assert (
            elems <= self.feeds["cardinality"]
        ), f"{elems} to take greater than cardinality of: {self.feeds['cardinality']}"

        def get_indices(*args):
            return self.feeds["indices"](*args)[:elems]

        return Dataset(
            {
                "cardinality": elems,
                "reader": self.feeds["reader"],
                "tree_def": self.feeds["tree_def"],
                "indices": get_indices,
            }
        )

    def skip(self, elems: int) -> "Dataset":
        """Skips the first `elems` of the dataset.

        Parameters
        ----------
        elems : int
            desired number of observations to skip.

        Returns
        -------
        Dataset
            Skipped Dataset.
        """
        assert (
            elems <= self.feeds["cardinality"]
        ), f"{elems} to skip greater than cardinality of: {self.feeds['cardinality']}"

        def get_indices(*args):
            return self.feeds["indices"](*args)[elems:]

        return Dataset(
            {
                "cardinality": self.feeds["cardinality"] - elems,
                "reader": self.feeds["reader"],
                "tree_def": self.feeds["tree_def"],
                "indices": get_indices,
            }
        )

    def shuffle(self, seed:DeviceArray) -> "Dataset":
        """Shuffles the observations based on `seed`.

        Parameters
        ----------
        seed:  DeviceArray
            Random key to ensure reproducibility.

        Returns
        -------
        Dataset
            Shuffled Dataset.
        """
        def get_indices(ix):
            return permutation(seed+ix, self.feeds["indices"](ix), independent=True)

        return Dataset(
            {
                "cardinality": self.feeds["cardinality"],
                "reader": self.feeds["reader"],
                "tree_def": self.feeds["tree_def"],
                "indices": get_indices,
            }
        )

    def batch(self, batch_size: int, drop_remainder: bool = False) -> Any:
        """Generator to create batches given a desired `batch_size`.

        Parameters
        ----------
        batch_size : int
            Desired batch size.
        drop_remainder : bool, optional
            Whether to skip remainder data at the end of the epoch, by default False

        Returns
        -------
        Any
            A pytree of the batched data.

        Yields
        ------
        Iterator[Any]
            Iterator of pytrees of the batched data.
        """
        reader = self.feeds["reader"]
        indices = self.feeds["indices"](self.epoch)
        cardinality=self.feeds["cardinality"]

        batch_schedule,n_batches=get_batch_schedule(batch_size,indices,cardinality)

        for batch_index in range(n_batches):
            yield reader(nested_indexer(batch_index, batch_schedule, indices))

        if not (drop_remainder) and self.feeds["cardinality"] % batch_size != 0:
            yield reader(indices[n_batches * batch_size :])
        
        self.epoch+=1

    def cardinality(self) -> int:
        """Returns the number of available samples in the dataset.

        Returns
        -------
        int
            The number of available samples in the dataset.
        """
        return self.feeds["cardinality"]

    def jit(self, warmup:bool=True) -> "Dataset":
        """Jits any mapping operation previouslyapplied to the dataset.

        Consoder to use `jit` after one or more `map` calls. 

        Parameters
        ----------
        warmup : bool, optional
            If True pre-compiles previous `map` operations, otherwise
            pre-compilation is left at runtime, by default True

        Returns
        -------
        Dataset
            Jitted Dataset.
        """
        reader = jit(self.feeds["reader"], backend="cpu")
        if warmup:
            _ = reader(jnp.arange(2))
            _ = nested_indexer(0, jnp.arange(10), self.feeds["indices"](0))

        return Dataset(
            {
                "cardinality": self.feeds["cardinality"],
                "reader": reader,
                "tree_def": self.feeds["tree_def"],
                "indices": self.feeds["indices"],
            },
        )
    
    def reset(self):
        """Resets the internal random number generator.
        """
        self.epoch=0

    @staticmethod
    def from_tensor_slices(tensors: Any) -> "Dataset":
        """Generates a Dataset from any input tensors.

        The only supported way to generate a `snax.data.Dataset` for now.
        It will be extened in the future to also include the equivalent of `tf.data.Dataset.from_generator`.
        Behavior is similar to `tf.data.Dataset.from_tensor_slices`.

        Parameters
        ----------
        tensors : Any
            Any pytree.

        Returns
        -------
        Dataset
            Snax Dataset
        """
        tensors = to_jax_pytree(tensors)
        tree_def = tree_structure(tensors)

        def cardinality():
            leaves_cardinalities = leaves_sizes(tensors)
            assert (
                len(set(leaves_cardinalities)) == 1
            ), f"Leaves must all share the same cardinality but found: {build_tree(tree_def,leaves_cardinalities)}"
            return leaves_cardinalities[0]

        def get_indices(*args):
            return jnp.arange(cardinality())

        def reader(indices):
            return index_elems_in_tree(indices, tensors)

        return Dataset(
            {
                "cardinality": cardinality(),
                "reader": reader,
                "tree_def": tree_def,
                "indices": get_indices,
            }
        )

    @staticmethod
    def zip(*datasets: Iterable["Dataset"]) -> "Dataset":
        """Zips two or more `snax.data.Dataset`s together.

        All `datasets` should have the same cardinality.


        Returns
        -------
        Dataset
            Zipped Dataset
        """
        cardinalities = set([dataset.cardinality() for dataset in datasets])
        assert (
            len(cardinalities) == 1
        ), f"Datasets must all have the same cardinality, found: {cardinalities}"

        feeds = [dataset.feeds for dataset in datasets]
        cardinality = list(cardinalities)[0]
        merger_indices = jnp.arange(cardinality)

        def reader(indices):
            pointers = merger_indices[indices]
            data = []
            for feed in feeds:
                zipped_indices = feed["indices"](0)[pointers]
                data=data+tree_leaves(feed["reader"](zipped_indices))
            return data

        def get_indices(*args):
            return jnp.arange(cardinality)

        return Dataset(
            {
                "cardinality": cardinality,
                "reader": reader,
                "tree_def": tree_structure(reader(jnp.arange(2))),
                "indices": get_indices,
            }
        )
