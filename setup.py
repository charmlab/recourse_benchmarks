import pathlib
import re

from setuptools import find_packages, setup

VERSIONFILE = "_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    VERSION = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="recourse-benchmarks",
    version=VERSION,
    description="A library for counterfactual recourse",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/charmlab/recourse_benchmarks",
    author="Amir-Hossein Karimi, Corinna Coupette, Abubakar Bello",
    author_email="amirh.karimi@uwaterloo.ca",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=("test",)),
    include_package_data=True,
    install_requires=[
        "importlib_resources",
        "black",
        "causalgraphicalmodels==0.0.4",
        "Flask==1.1.2",
        "dash==2.15.0",
        "dice-ml==0.5",
        "flake8",
        "h5py==2.10.0",
        "imageio==2.9.0",
        "ipython==7.16.0",
        "jinja2==2.11.3",
        "keras==2.3.0",
        "lime==0.2.0.1",
        "mip==1.12.0",
        "networkx==2.5.1",
        "numpydoc==1.1.0",
        "numpy==1.19.4",
        "markupsafe==2.0.1",
        "itsdangerous==2.0.1",
        "werkzeug==2.0.3",
        "pandas==1.1.4",
        "pre-commit==2.9.2",
        "protobuf<=3.21",
        "PySMT==0.9.5",
        "pytest==6.1.2",
        "recourse==1.0.0",
        "scikit-learn==0.23.2",
        "scipy==1.6.2",
        "sphinx==4.0.2",
        "sphinx_autodoc_typehints==1.12.0",
        "sphinx-rtd-theme==0.5.2",
        "tensorflow==1.14.0",
        "gast==0.2.2",
        "torch==1.7.0",
        "torchvision==0.8.1",
        "xgboost==1.4.2",
        "tqdm",
    ],
)
