from setuptools import setup

setup(
    name='lung_map_utils',
    version='0.1dev',
    packages=['lung_map_utils', ],
    license='BSD 2-Clause License',
    long_description=open('README.md').read(),
    author='Duke Lungmap Team',
    description='A machine learning pipeline used to predict anatomy from Lungmap images.',
    install_requires=[
        'numpy==1.13.1',
        'opencv-python==3.2.0.7',
        'matplotlib==2.0.2',
        'scipy==0.19.1',
        'pandas==0.19.2',
        'scikit-learn==0.18.2'],
)
