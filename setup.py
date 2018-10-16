import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ballistico",
    version="0.0.1",
    author="Giuseppe Barbalinardo",
    author_email="giuseppe.barbalinardo@gmail.com",
    description="Ballistico Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/gbarbalinardo/ballistico-new",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)