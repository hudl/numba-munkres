from setuptools import setup, find_packages


setup(
    name='munkres',
    version='1.1.0',
    description='munkres algorithm for the assignment problem',
    packages=find_packages(),
    install_requires=['numba', 'numpy'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
