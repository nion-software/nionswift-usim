# -*- coding: utf-8 -*-

"""
To upload to PyPI, PyPI test, or a local server:
python setup.py bdist_wheel upload -r <server_identifier>
"""

import setuptools
import os

setuptools.setup(
    name="nionswift-usim",
    version="0.1.7",
    author="Nion Software",
    author_email="swift@nion.com",
    description="A Nion Swift package to simulate a STEM microscope, scanner, and cameras",
    long_description=open("README.rst").read(),
    url="https://github.com/nion-software/nionswift-usim",
    packages=["nionswift_plugin.usim", "nionswift_plugin.usim.test"],
    package_data={"nionswift_plugin.usim": ["manifest.json"]},
    install_requires=['nionswift-instrumentation'],
    license='GPLv3',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.6",
    ],
    include_package_data=True,
    python_requires='~=3.6',
)
