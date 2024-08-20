from setuptools import setup, find_packages

setup(
    name='liver_annotation',
    version='0.1.1',
    author='Madhavendra Thakur',
    author_email='madhavendra.thakur@gmail.com',
    description='A machine learning model for classification of cells and annotation of clusters in scRNA-seq data from liver samples.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mThkTrn/liver_annotation',
    packages=find_packages(),
    install_requires=[
        'joblib>=1.4.2',
        'torch>=2.1.2+cpu',
        'scikit-learn>=1.2.2',
        'scipy>=1.11.4'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)