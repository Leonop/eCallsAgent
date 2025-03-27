from setuptools import setup, find_packages

setup(
    name="eCallsAgent",
    version="0.1.0",
    author="Zicheng Xiao",
    description="Package for analyzing earnings call transcripts using LLM and topic modeling",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'umap-learn',
        'matplotlib',
        'seaborn',
        'transformers',
        'torch',
        # Add any other dependencies
    ],
    python_requires='>=3.8',
) 