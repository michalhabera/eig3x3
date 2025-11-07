# Have to keep the setup.py due to incompatible CFFI build
# step and pyproject.toml setup.
from setuptools import setup

setup(
    cffi_modules=["src/eig3x3/_build_cffi.py:ffibuilder"],
    packages=["eig3x3"],
    package_dir={"eig3x3": "src"},
    package_data={"eig3x3": ["c/eig3x3.h"]},
)
