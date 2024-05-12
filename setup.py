from setuptools import setup, find_packages

setup(
    name='SelfCalibratedConformal',
    version='0.3',
    packages=find_packages(),
    description='A Python implementation of Self-Calibrated Conformal Prediction',
    long_description='A Python implementation of Self-Calibrated Conformal Prediction',
    long_description_content_type='text/markdown',  # assuming your description is in Markdown format
    author='Lars van der Laan',
    author_email='vanderlaanlars@yahoo.com',
    url='https://github.com/Larsvanderlaan/SelfCalibratedConformal',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'statsmodels',
        # Correct: 'math' module is part of Python's standard library and not installable via pip, so it should not be listed here.
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
