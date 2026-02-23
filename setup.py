import os
from setuptools import setup, find_packages

package_name = "flownets"

# Directory dove si trova questo setup.py
HERE = os.path.abspath(os.path.dirname(__file__))

def read_requirements():
    req_path = os.path.join(HERE, "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, encoding="utf-8") as f:
            return f.read().splitlines()
    return []  # non far fallire la build

# Legge la versione
version_py = os.path.join(HERE, package_name, "version.py")
with open(version_py, encoding="utf-8") as f:
    version = f.read().split(" ")[-1][1:-1]

# Legge il README
readme_path = os.path.join(HERE, package_name, "README.md")
with open(readme_path, encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=package_name,
    version=version,
    description="A new beginning for my models :)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TommyGiak/FlowNets",
    author="Tommaso Giacometti",
    author_email="tommaso.giak@gmail.com",
    license="MIT",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)