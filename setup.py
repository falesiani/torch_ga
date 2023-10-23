from setuptools import setup
import os

# Read README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as readme_file:
    long_description = readme_file.read()

setup(
    name="torch_ga",
    packages=["torch_ga"],
    extras_require={
        "torch": ["pytorch>=1.12.1"],
    },
    description="Clifford and Geometric Algebra with PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.0.2",
    url="https://github.com/falesiani/torch_ga",
    author="Francesco Alesiani",
    author_email="francesco.alesiani@neclab.eu",
    license="MIT",
    keywords="geometric-algebra clifford-algebra pytorch multi-vector para-vector mathematics machine-learning",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",

        "Programming Language :: Python :: 3",

        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",

        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",

        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ]
)
