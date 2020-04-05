import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nox-if",
    version="0.0.12",
    author="Renos Zabounidis",
    author_email="rzabounidis@cs.umass.edu",
    description="NoX Normalizing Flow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Information-Fusion-Lab-Umass/NoX",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)