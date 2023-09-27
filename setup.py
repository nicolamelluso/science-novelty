from setuptools import setup, find_packages

setup(
    name="science_novelty",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tqdm',
        'pandas'
    ],
)