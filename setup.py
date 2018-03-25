#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

requirements = [
    'keras',
    'cairocffi',
    'scikit-image',
    'pandas',
    'backports.weakref',
    'scikit-learn',
    'h5py',
    'tensorflow',
    'editdistance'
    ]

setup_args = {
    'name': 'NRC OCR',
    'version': '0.1.0',
    'description': "OCR for digitizing NRC id numbers",
    'long_description': "OCR for digitizing NRC id numbers",
    'author': "Thuso 'Danger' Simon",
    'author_email': 'thuso@ilovezoona.com',
    'url': '',
    'packages': find_packages(),
    'package_dir': {'nrc_ocr': 'nrc_ocr'},
    'install_requires': requirements,
    'setup_requires': ['pytest-runner'],
    'tests_require': ['pytest', 'pytest-mock'],
    'keywords': 'OCR',
    'classifiers': [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Zoona Data Team',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ]}

setup(**setup_args)
