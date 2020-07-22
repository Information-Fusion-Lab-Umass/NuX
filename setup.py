import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nux", # Replace with your own username
    version="1.0.2",
    author="Information Fusion Lab",
    author_email="rzabounidis@cs.umass.edu",
    description="Normalizing Flows using Jax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Information-Fusion-Lab-Umass/NuX",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy >=1.12', 'jax','jaxlib'
    ]
)