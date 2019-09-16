from setuptools import setup
import glob
import os

setup(
    name="dist_py",
    version="0.0.2",
    author="Kartheik Iyer",
    author_email="iyer@physics.rutgers.edu",
    url = "https://github.com/kartheikiyer/dist_py",
    packages=["dist_py"],
    description="Statistical Tests Comparing distributions in > 1 Dimensions",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=["matplotlib", "numpy", "scipy"]
)
