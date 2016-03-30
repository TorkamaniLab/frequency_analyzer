import re
from setuptools import setup, find_packages


setup(
    name='frequency_analyzer',
    version='1.0',
    packages=find_packages(),
    description='A command-line tool for isolating frequencies found in accelerometer data.',
    url='https://github.com/TorkamaniLab/frequency_analyzer',
    entry_points = {
        "console_scripts": ['frequency_analyzer = frequency_analyzer.app:main']
        },
    install_requires = ['PyWavelets']
    author='Brian Schrader',
    author_email='brian@brianschrader.com',
)
