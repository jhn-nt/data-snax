from pathlib import Path
import jax.numpy as jnp
import numpy as np
import jax.random as jrn
from jax import vmap
from jaxtyping import PyTree, Int
from typing import Callable, Union, List, Iterable, Any
from .tree_util import *


def filter_and_flatten(schedule):
    schedule_batches = [batch for batch in schedule]

    def filter_and_flatten_batch(batch):
        batch = jnp.reshape(batch, (-1,))
        batch = batch[jnp.isfinite(batch)].astype(jnp.int32)
        return batch

    return list(map(filter_and_flatten_batch, schedule_batches))


class Dataset:
    def __init__(
        self,
        reader: Callable[[Union[int, Array]], PyTree],
        sizer: Callable[[], int],
        scheduler: Callable[[int], List[Union[Int[Array, "1 dim"], Int]]],
    ):
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
        if self.not_ready:
            self.setup()

        if self.step < self.batches_in_epoch:
            return self.return_current_batch_and_index_next_batch()
        else:
            self.reset_epoch_and_update_rng()
            raise StopIteration

    def __iter__(self):
        return self

    def setup(self):
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
        self.rng = self.ref
        self.step = 0

    def batch(self, batch_size: int, drop_reminder: bool = False) -> "Dataset":
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
        def scheduler(rng):
            return jrn.permutation(seed + rng, self.scheduler(rng))

        return Dataset(reader=self.reader, sizer=self.sizer, scheduler=scheduler)

    def map(self, func: Callable) -> "Dataset":
        def reader(ix):
            return vmap(func)(self.reader(ix))

        return Dataset(reader=reader, sizer=self.sizer, scheduler=self.scheduler)

    def jit(self) -> "Dataset":
        reader = jit(self.reader, backend="cpu")
        return Dataset(reader=reader, sizer=self.sizer, scheduler=self.scheduler)

    def take(self, elems: int) -> "Dataset":
        def sizer():
            return elems

        def scheduler(rng):
            return self.scheduler(rng)[:elems]

        return Dataset(reader=self.reader, sizer=sizer, scheduler=scheduler)

    def skip(self, elems: int) -> "Dataset":
        def sizer():
            return self.sizer() - elems

        def scheduler(rng):
            return self.scheduler(rng)[elems:]

        return Dataset(reader=self.reader, sizer=sizer, scheduler=scheduler)

    def apply(self, func: Callable[[Any], "Dataset"]) -> "Dataset":
        return func(self)

    @staticmethod
    def from_tensor_slices(tensors: PyTree) -> "Dataset":
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
        sizes = list(set(list(map(lambda x: x.sizer(), datasets))))
        assert (
            len(sizes) == 1
        ), f"All datasets must have the same cardinality, found: {sizes}"

        _ = list(map(lambda x: x.setup(), datasets))

        def sizer():
            return sizes[0]

        def reader(ix):
            output = []
            for dataset in datasets:
                _ix = jnp.array([dataset._schedule[i] for i in ix])
                output += tree_leaves(dataset.reader(_ix))
            return output

        schedule = jnp.transpose(jnp.atleast_2d(jnp.arange(sizer())))

        def scheduler(rng):
            return schedule

        return Dataset(reader=reader, sizer=sizer, scheduler=scheduler)

    @staticmethod
    def concatenate(self, *datasets: Iterable["Dataset"]) -> "Dataset":
        pass

    @staticmethod
    def from_generator(self, gererator, elems: int) -> "Dataset":
        def sizer():
            return elems
