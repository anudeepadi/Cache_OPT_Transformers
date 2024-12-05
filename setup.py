from setuptools import setup, find_packages

setup(
    name="cache_opt_transformer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "psutil>=5.8.0",
        "torch>=1.9.0",
        "matplotlib>=3.4.3",
        "pandas>=1.3.0",
    ],
    author="Venkata Anudeep Adiraju",
    author_email="venkataanudeep.adiraju@utsa.edu",
    description="CPU Cache Optimization for Small-Scale Transformer Models",
    python_requires=">=3.7",
)