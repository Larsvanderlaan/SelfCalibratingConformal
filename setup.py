from setuptools import setup, find_packages

setup(
    name='SelfCalibratedConformal',
    version='0.1',
    packages=find_packages(),
    description='A Python implementation of Self-Calibrated Conformal Prediction',
    author='Lars van der Laan',
    author_email='vanderlaanlars@yahoo.com',
    url='https://github.com/Larsvanderlaan/SelfCalibratedConformal',
    install_requires=[
        # List your package dependencies here
        'numpy',  # For example, if your package uses NumPy
    ],
)
