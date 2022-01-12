from setuptools import setup, find_packages

with open("MolAligner/version.py") as FH:
    __version__ = FH.read().split("=")[-1]

print(find_packages())
setup(
    name="MolAligner",
    version=__version__,
    description="A toolkit for molecular geometry manipulation.",
    url="https://github.com/masrul/MolAligner",
    author="Masrul Huda",
    author_email="mmh568@msstate.edu",
    packages=["MolAligner"],
    install_requires=["numpy>=1.14"],
    python_requires=">=3.7",
    classifiers=[
        "Intended Audience :: Molecular Simulation",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
    ],
)
