import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


import snax.datasets as snds
import jax.numpy as jnp
import flax.linen as nn

from jax.random import PRNGKey

W = jnp.ones((784, 256))


def processing(input):
    image = input["image"]
    x = image.astype("float32") / 255
    x = jnp.reshape(x, (-1,))
    return {"image": nn.relu(jnp.matmul(x, W)), "label": input["label"]}


van_snax = (
    snds.load("mnist")["train"].map(processing).shuffle(PRNGKey(0)).jit().batch(128)
)

for _ in range(5):
    for batch in van_snax:
        batch
