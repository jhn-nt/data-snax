from setuptools import setup


setup(
    name="data-snax",
    version="0.2",
    description="Versatile data ingestion pipelines for jax",
    install_requires=[
        "numpy>=1.23",
        "jax>=0.3.25",
        "requests>=2.28.1",
        "h5py>=3.7.0",
        "jaxtyping==0.2.11",
    ],
    license="Apache 2.0",
    author="jhn-nt",
    packages=["snax"],
    package_data={"snax": ["datasets/*.json"]},
)
