![](ts_logo_blue.png)

# Welcome to TokSearch

TokSearch is a Python package for parallel retrieving, processing, and filtering of arbitrary-dimension fusion experimental data. TokSearch provides a high level API for extracting information from many shots, along with useful classes for low level data retrieval and manipulation.

The fundamental class in TokSearch is the ```Pipeline```. A ```Pipeline``` object takes a list of shots and, for each shot in the list, creates a dict-like object called a ```Record```. The ```Pipeline``` object then provides methods for defining a sequence of processing steps to apply to each record. These processing steps include:

- Passing user-defined functions to the pipeline via the ```map``` method.

or...

- Using a set of built-in methods, such as ```fetch```, ```fetch_dataset```, ```align```, ```keep```, or ```discard```.

The ```Pipeline``` also provides a ```where``` method which takes as input a user-defined function that returns a boolean value. If the function evaluates to ```False``` for a record, then that record is removed from the pipeline.

## Installation

At the moment, the cleanest way to install TokSearch is to first set up a Conda/Mamba environment with the required dependencies, and then install TokSearch from the local clone of the repository. Here are the steps:

First, clone the repository, then from the root directory of the repository, run:

```bash
mamba env create -f environment.yml
```

or

```bash
conda env create -f environment.yml
```

You can also specify the ```-p``` flag to specify the path to the environment. For example:

```bash
mamba env create -f environment.yml -p /path/to/env
```

Then, activate the environment:

```bash
conda activate toksearch # or whatever you named the environment
```

Finally, install TokSearch itself:

```bash
pip install -e .
```
## Viewing the Documentation

Eventually, the documentation will be hosted on Github Pages. For now, you can build the documentation locally by running:

```bash
mkdocs serve
```

The documentation will be available at ```http://localhost:8000``` (or possibly a different port if 8000 is already in use).
