from pathlib import Path
import jax.numpy as jnp
import numpy as np
import jax.random as jrn
from jax import vmap
from jaxtyping import PyTree, Int
from typing import Callable, Union, List, Iterable, Any
from .tree_util import *


def filter_and_flatten(
    schedule: Int[Array, "n-batches"]
) -> List[Int[Array, "batch_size"]]:
    """Given a matrix of float 'schedule' it returns a list of length equal to the first dimension of schedule where each elements represent the flattened dimensions of the input"""

    def filter_and_flatten_batch(batch):
        batch = jnp.ravel(batch)
        batch = batch.take(~jnp.isnan(batch)).astype(jnp.int32)
        return batch

    if len(schedule.shape) == 2:
        # for performance, unbatched sets are processed as a single large batch and then split
        output = list(filter_and_flatten_batch(schedule))
    else:
        schedule_batches = list(schedule)
        output = tree_map(filter_and_flatten_batch, schedule_batches)

    return output


def assert_same_cardinality_across_datasets(datasets: List["Dataset"]) -> int:
    """Asserts that all Datasets have same cardinality and returns it.

    Parameters
    ----------
    datasets : List[Dataset]
        Iterable of datasets.

    Returns
    -------
    int
        Cardinality of the datasets.
    """
    sizes = list(set(list(map(lambda x: x.sizer(), datasets))))
    assert (
        len(sizes) == 1
    ), f"All datasets must have the same cardinality, found: {sizes}"
    return sizes[0]


def return_currentmost_schedule_as_array(dataset: "Dataset") -> Int[Array, "n-batches"]:
    """Return the currently scheduled indexes of the batch of 'dataset'.

    Returns
    -------
    _type_
        Indexes of the current batch.
    """
    return dataset.scheduler(dataset.rng)


class Dataset:
    def __init__(
        self,
        reader: Callable[[Union[int, Array]], PyTree],
        sizer: Callable[[], int],
        scheduler: Callable[[int], List[Union[Int[Array, "1 dim"], Int]]],
    ):
        """A jax interpretation of tensorflow `tf.data.Dataset`.
        In this current verions it supports:
        #. `map`: similar behavior to  `tf.data.Dataset.map`
        #. `take`: similar behavior to  `tf.data.Dataset.take`
        #. `skip`: similar behavior to  `tf.data.Dataset.skip`
        #. `zip`: similar behavior to  `tf.data.Dataset.zip`
        #. `shuffle`: behavior is equal to `tf.data.Dataset.shuffle` but with different input signature,
            instead of a `shuffle_buffer` it requires a `jax.PRNGKEY`.
        #. `batch`: similar behavior to `tf.data.Dataset.batch`
        #. `apply`: similar behavior to `tf.data.Dataset.apply`
        What it does not yet support:
        #. `from_generator`
        #. `concatenate`
        #. `bucket_by_sequence_lenght`
        #. `filter`

        """
        self.reader = reader
        self.sizer = sizer
        self.scheduler = scheduler

        # state variables
        self.step = 0
        self.rng = jnp.array(0)
        self.inc = jnp.array(1)
        self.ref = jnp.array(0)
        self.not_ready = True

    def __next__(self):
        if self.is_step_lower_than_epoch():
            return self.return_current_batch_and_index_next_batch()
        else:
            self.reset_epoch_and_update_rng()
            raise StopIteration

    def __iter__(self):
        self.setup_if_not_ready()
        return self

    def is_step_lower_than_epoch(self):
        return self.step < self.batches_in_epoch

    def setup_if_not_ready(self):
        if self.not_ready:
            self._schedule = filter_and_flatten(self.scheduler(self.rng))
            self.batches_in_epoch = self.sizer()
            self.not_ready = False

    def reset_epoch_and_update_rng(self):
        self.step = 0
        self.rng += self.inc
        self._schedule = filter_and_flatten(self.scheduler(self.rng))

    def return_current_batch_and_index_next_batch(self):
        output = self.reader(self._schedule[self.step])
        self.step += 1
        return output

    def reset(self):
        """Resets the internal state of the Dataset."""
        self.rng = self.ref
        self.step = 0

    def batch(self, batch_size: int, drop_reminder: bool = False) -> "Dataset":
        """Batches the dataset.

        Parameters
        ----------
        batch_size : int
            Desired batch size.
        drop_reminder : bool, optional
            Whether to skip remainder data at the end of the epoch, by default False

        Returns
        -------
        Dataset
            Batched Dataset.
        """
        n_batches = self.sizer() // batch_size
        extra_samples = batch_size - (self.sizer() % batch_size)
        n_samples = n_batches * batch_size

        def sizer():
            return n_batches if drop_reminder else n_batches + 1

        def scheduler(rng):
            schedule = self.scheduler(rng)
            if drop_reminder:
                schedule = jnp.concatenate(
                    [
                        schedule,
                    ]
                )
            batches = jnp.reshape(
                schedule[:n_samples],
                (n_batches, batch_size, *schedule.shape[1:]),
            )

            if not drop_reminder:
                padding = jnp.nan * jnp.empty((extra_samples, *schedule.shape[1:]))
                padding = jnp.expand_dims(
                    jnp.concatenate([schedule[n_samples:], padding], axis=0), axis=0
                )
                batches = jnp.concatenate([batches, padding], axis=0)

            return batches

        return Dataset(reader=self.reader, sizer=sizer, scheduler=scheduler)

    def shuffle(self, seed: Array) -> "Dataset":
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

        def scheduler(rng):
            return jrn.permutation(seed + rng, self.scheduler(rng))

        return Dataset(reader=self.reader, sizer=self.sizer, scheduler=scheduler)

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

        def reader(ix):
            return vmap(func)(self.reader(ix))

        return Dataset(reader=reader, sizer=self.sizer, scheduler=self.scheduler)

    def jit(self, warmup: bool = True) -> "Dataset":
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
        reader = jit(self.reader, backend="cpu")

        if warmup:
            _ = reader(self.scheduler(0))
        return Dataset(reader=reader, sizer=self.sizer, scheduler=self.scheduler)

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

        def sizer():
            return elems

        def scheduler(rng):
            return self.scheduler(rng)[:elems]

        return Dataset(reader=self.reader, sizer=sizer, scheduler=scheduler)

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

        def sizer():
            return self.sizer() - elems

        def scheduler(rng):
            return self.scheduler(rng)[elems:]

        return Dataset(reader=self.reader, sizer=sizer, scheduler=scheduler)

    def apply(self, func: Callable[[Any], "Dataset"]) -> "Dataset":
        """Applies `func` to the `snax.data.Dataset`.

        `func` should return a `snax.data.Dataset` object.


        Parameters
        ----------
        func : Callable
            Desired transofrmation to apply to the dataset.
        Returns
        -------
        Dataset
            Mapped dataset.
        """
        return func(self)

    def cardinality(self) -> int:
        """Returns the cardinality of the dataset.

        Returns
        -------
        int
            Cardinality of the dataset.
        """
        return self.sizer()

    def filter(self, predicate: Callable[[Any], bool]) -> "Dataset":
        status_set = Dataset.zip(self, self.map(predicate))
        return status_set

    @staticmethod
    def from_tensor_slices(tensors: PyTree) -> "Dataset":
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

        def sizer():
            return tree_height(tensors)

        def reader(ix):
            return tree_index(tensors, ix)

        schedule = jnp.transpose(jnp.atleast_2d(jnp.arange(sizer())))

        def scheduler(rng):
            return schedule

        return Dataset(reader=reader, sizer=sizer, scheduler=scheduler)

    @staticmethod
    def zip(*datasets: Iterable["Dataset"]) -> "Dataset":
        """Zips two or more `snax.data.Dataset`s together.
        All `datasets` should have the same cardinality.

        Parameters
        ----------
        datasets : Iterable["Dataset"]
            An iterable of datasets to zip.
        Returns
        -------
        Dataset
            Zipped Dataset
        """
        size = assert_same_cardinality_across_datasets(datasets)
        root_schedules = list(map(return_currentmost_schedule_as_array, datasets))

        def sizer():
            return size

        def reader(ix):
            return list(
                map(lambda x: x[0].reader(x[1][ix]), zip(datasets, root_schedules))
            )

        schedule = jnp.transpose(jnp.atleast_2d(jnp.arange(sizer())))

        def scheduler(rng):
            return schedule

        return Dataset(reader=reader, sizer=sizer, scheduler=scheduler)

    @staticmethod
    def concatenate(*datasets: Iterable["Dataset"]) -> "Dataset":
        sizes = tree_map(lambda x: x.sizer(), datasets)
        root_schedule = jnp.hstack(
            tree_map(return_currentmost_schedule_as_array, datasets)
        )

        cumulative_size = jnp.asarray(sum(sizes))

        def sizer():
            return cumulative_size

        def reader(ix):
            nested_ix = root_schedule.take(ix)
            # nested_readers = root_readers.take(ix)
            pass

        schedule = jnp.transpose(jnp.atleast_2d(jnp.arange(sizer())))

        def scheduler(rng):
            return schedule
