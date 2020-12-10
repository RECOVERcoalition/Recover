from setuptools import setup, find_packages

"""
The first time you need to run:
 pip install -e .
"""

setup(
    name='recover',
    description='Exploration of drug combinations using active learning',
    packages=find_packages(include=["recover", "recover.*"]),
    version="0.1")
