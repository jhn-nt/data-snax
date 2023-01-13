import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import snax.datasets as snds
import snax.data as sn
from jax.random import PRNGKey
from tqdm import tqdm

ds = snds.load("mnist")["train"]

ds_1 = ds.take(30000)
ds_2 = ds.skip(30000)

ds = sn.Dataset.zip(ds_1, ds_2).shuffle(PRNGKey(0))

for batch in tqdm(ds.batch(128)):
    batch
