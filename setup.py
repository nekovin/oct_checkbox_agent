"""Setup script for OCR Checkbox Agent."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "ocr_checkbox_agent" / "README.md").read_text()

# Read requirements
requirements = []
requirements_file = this_directory / "ocr_checkbox_agent" / "requirements.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="ocr-checkbox-agent",
    version="1.0.0",
    author="OCR Checkbox Agent Team",
    author_email="contact@example.com",
    description="Extract checkbox data from PDFs and convert to structured format",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/ocr-checkbox-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Text Processing :: Markup",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ocr-checkbox-agent=ocr_checkbox_agent.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ocr_checkbox_agent": [
            "templates/*.json",
            "sample_data/*.json",
            "examples/*.py",
        ],
    },
    keywords="ocr checkbox pdf extraction computer-vision tesseract",
    project_urls={
        "Bug Reports": "https://github.com/example/ocr-checkbox-agent/issues",
        "Source": "https://github.com/example/ocr-checkbox-agent",
        "Documentation": "https://ocr-checkbox-agent.readthedocs.io/",
    },
)