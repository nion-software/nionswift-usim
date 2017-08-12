# -*- coding: utf-8 -*-

"""
To upload to PyPI, PyPI test, or a local server:
python setup.py bdist_wheel upload -r <server_identifier>
"""

import setuptools
import os

setuptools.setup(
    name="nionswift-usim",
    version="0.0.1",
    author="Nion Software",
    author_email="swift@nion.com",
    description="Simulate a STEM microscope, scanner, and cameras",
    url="https://github.com/nion-software/nionswift-usim",
    packages=["nionswift_plugin.usim", ],
    license='GPLv3',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.5",
    ],
    include_package_data=True,
    python_requires='~=3.5',
)
