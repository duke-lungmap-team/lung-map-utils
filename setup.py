from setuptools import setup

setup(
    name='lung_map_utils',
    version='0.1dev',
    packages=['lung_map_utils', ],
    license='BSD 2-Clause License',
    long_description=open('README.md').read(),
    author='Duke Lungmap Team',
    description='A machine learning pipeline used to predict anatomy from Lungmap images.',
    install_requires=['numpy', 'opencv-python', 'matplotlib', 'scipy', 'pandas', 'scikit-learn'],
)
