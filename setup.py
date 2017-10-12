from setuptools import setup

setup(
    name='LungMapUtils',
    version='1.0',
    packages=['lung_map_utils'],
    license='BSD 2-Clause License',
    long_description=open('README.md').read(),
    author='Duke Lungmap Team',
    description='A machine learning pipeline used to predict anatomy from Lungmap images.',
    install_requires=[
        'numpy (==1.13)',
        'opencv_python (==3.2.0)',
        'scipy (==0.19.1)',
        'pandas (==0.19.2)',
        'scikit_learn (==0.18.2)'
    ]
)
