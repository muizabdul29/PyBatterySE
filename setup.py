"""Setup for the PyBatterySE package"""

# pylint: disable=E0401
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    readme = f.read()

setup(
    name='pybatteryse',
    version='2.0.1',
    author='Muiz Sheikh',
    description='Battery State Estimation in Python',
    long_description=readme,
    long_description_content_type='text/markdown',
    url="https://github.com/muizabdul29/PyBatterySE",
    packages=find_packages(include=['pybatteryse', 'pybatteryse.*']),
    install_requires=[
        'pybatteryid>=3.0.2',
        'numpy>=2.1.0',
        'tqdm>=4.67.3',
        'scipy<1.17'
    ],
    tests_require=[],
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
