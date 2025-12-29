from setuptools import setup, find_packages
import os

# 读取版本信息
def get_version():
    version_file = os.path.join("rlgym2d", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rlgym2d",
    version=get_version(),
    author="RLGym2D Contributors",
    description="RLGym2D - 2D强化学习仿真平台",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/RLGym2D",
    packages=find_packages(),
    package_data={
        "rlgym2d": ["**/*.py"],
    },
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rlgym2d-train=train:main",
            "rlgym2d-play=play:main",
        ],
    },
)