from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='hydrocode',
    version='0.1',
    description="A GPU-accelerated hydrodynamics code written in JAX.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url="https://github.com/leobetancourt/hydro-code",
    author="Leo Betancourt",
    author_email="leo.kbet@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'Click',
        'numpy',
        'jax',
        'jaxlib',
        'matplotlib',
        'scipy',
        'tqdm',
        'rich',
        'pandas',
        'h5py'
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'hydrocode=hydrocode.cli:cli'
        ]
    }
)