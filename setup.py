from setuptools import setup, find_packages


setup(
    name='numba_munkres',
    version='1.1.0',
    description='numba_munkres algorithm for the assignment problem',
    packages=find_packages(),
    install_requires=['numba', 'numpy'],
)
