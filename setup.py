from setuptools import setup

setup(
    name='cv_color_features',
    version='1.2b',
    packages=['cv_color_features'],
    license='BSD 2-Clause License',
    long_description=open('README.md').read(),
    author='Scott White',
    description='Utility library to generate feature metrics from regions in color images.',
    install_requires=[
        'numpy (>=1.13)',
        'opencv-python (>=3.2.0.7)',
        'scipy (>=0.19.1)',
        'pandas (>=0.19.2)'
    ]
)
