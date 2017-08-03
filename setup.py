from setuptools import find_packages, setup

setup(
    name='lung_map_utils',
    version='0.1dev',
    packages=['lung_map_utils',],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    author='Duke Lungmap Team',
    description=('A machine learning pipeline used to predict anatomy from Lungmap images.'),
    install_requires=['numpy', 'cv2', 'matplotlib', 'scipy', 'pandas', 're', 'scikit-learn'],
)
