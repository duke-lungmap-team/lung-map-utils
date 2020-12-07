from setuptools import setup

setup(
    name='cv_color_features',
    version='1.5.1',
    packages=['cv_color_features'],
    license='BSD 2-Clause License',
    long_description=open('README.md').read(),
    author='Scott White',
    description='Utility library to generate feature metrics from regions in color images.',
    install_requires=[
        'numpy (>=1.16)',
        'opencv-python (>=4.1)',
        'scipy (>=1.2)',
        'pandas (>=0.24)',
        'cv2_extras (>=0.5.4)'
    ]
)
