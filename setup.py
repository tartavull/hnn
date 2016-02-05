#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
   "tqdm>=3.7.1",
    "scikit-learn>=0.17",
    "sklearn>=0.0",
    "scipy>=0.17.0"
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='hnn',
    version='0.1.0',
    description="Hebbian Neural Network",
    long_description=readme + '\n\n' + history,
    author="Ignacio Tartavull",
    author_email='tartavull@princeton.edu',
    url='https://github.com/tartavull/hnn',
    packages=[
        'hnn',
    ],
    package_dir={'hnn':
                 'hnn'},
    include_package_data=True,
    install_requires=requirements,
    license="ISCL",
    zip_safe=False,
    keywords='hnn',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
