"""Setup script for Solana MCP Server."""

import os
from setuptools import setup, find_packages

# Get description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Get version from package
about = {}
with open(os.path.join("solana_mcp", "__init__.py"), "r", encoding="utf-8") as f:
    exec(f.read(), about)

# Get dependencies from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f.readlines() if not line.startswith("#")]

setup(
    name="solana-mcp-server",
    version=about["__version__"],
    description="A comprehensive Solana token analysis server with focus on pumpfun tokens",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=about["__author__"],
    author_email=about["__email__"],
    url="https://github.com/yourusername/solana-mcp-server",
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Framework :: FastAPI",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: Cryptocurrency :: Blockchain",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "solana-mcp=solana_mcp.__main__:main",
        ],
    },
) 