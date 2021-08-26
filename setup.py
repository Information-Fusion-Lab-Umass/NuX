import setuptools

setuptools.setup(
    name="nux",
    version="2.0.0",
    author="Information Fusion Lab",
    author_email="edmondcunnin@cs.umass.edu",
    description="Normalizing Flows using Jax",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Information-Fusion-Lab-Umass/NuX",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=open("requirements.txt").read().splitlines(),
)