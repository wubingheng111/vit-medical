from setuptools import setup, find_packages

setup(
    name='vit-medical',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for pre-training Vision Transformers for medical applications.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.7.0',
        'transformers>=4.0.0',
        'datasets>=1.0.0',
        'numpy',
        'pandas',
        'scikit-learn',
        'opencv-python',
        'PyYAML',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)