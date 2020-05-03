FROM jupyter/scipy-notebook:latest
USER root

# Install graphviz
RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz graphviz-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install the latest versions of these python packages
RUN python -m pip install --upgrade pip && \
    pip uninstall numpy -y && \
    pip install --user numpy scipy pandas bokeh cython networkx graphviz \
    pygraphviz PyQt5 matplotlib opt_einsum autograd pymc3 recordclass seaborn tqdm \
    tensorflow-datasets tensorflow

RUN pip install --upgrade jax jaxlib

WORKDIR /app

# Make jupyter act like sublime (https://forums.fast.ai/t/tip-using-sublime-text-editing-shortcuts-in-jupyter-notebook-cells/8259)
RUN mkdir /home/jovyan/.jupyter/custom
ADD custom.js /home/jovyan/.jupyter/custom/