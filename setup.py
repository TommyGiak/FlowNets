import os
from setuptools import setup, find_packages

package_name = "flownets"
HERE = os.path.dirname(os.path.abspath(__file__))

version_py = os.path.join(os.path.dirname(__file__), package_name, "version.py")
version = version = open(version_py).read().split(' ')[-1][1:-1]
requirements = open(os.path.join(HERE, "requirements.txt")).read().split("\n")

setup(
  name=package_name,
  version=version,
  description="A new beginning for my models :)",
  long_description = open(os.path.join(HERE, "flownets", "README.md"), encoding="utf-8").read(),
  long_description_content_type="text/markdown",
  url="https://github.com/TommyGiak/FlowNets",
  author="Tommaso Giacometti",
  author_email="tommaso.giak@gmail.com",
  license="MIT",
  packages=find_packages(exclude=("tests", "docs")),
  include_package_data=True,
  install_requires=requirements,
  python_requires=">=3.8",
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
)
