from typing import List

from setuptools import find_packages, setup


def get_requirements(file_path: str) -> List[str]:
    """Read requirements, skipping blank lines, comments and '-e .'."""
    with open(file_path) as f:
        lines = [line.strip() for line in f]
    return [line for line in lines if line and not line.startswith("#") and line != "-e ."]


setup(
    name="student-performance-prediction",
    version="0.1.0",
    author="Tejas Gahallot",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
