from setuptools import setup

setup(
    name="btv",
    version="0.1",
    description="An information-theoretic data pruning and selection utility",
    author="JSeaward",
    author_email="joeseaward@gmail.com",
    packages=["btv"],
    install_requires=[
        "scikit-learn>=1.3",
        "prettytable>1",
        "scipy>=0.12.0",
        "pandas",
        "openml",
        "tqdm",
    ],
)
