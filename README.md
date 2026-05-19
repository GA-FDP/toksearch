![](ts_logo_blue.png)

# Welcome to TokSearch

TokSearch is a Python package for parallel retrieving, processing, and filtering of arbitrary-dimension fusion experimental data. TokSearch provides a high level API for extracting information from many shots, along with useful classes for low level data retrieval and manipulation.

The fundamental class in TokSearch is the ```Pipeline```. A ```Pipeline``` object takes a list of shots and, for each shot in the list, creates a dict-like object called a ```Record```. The ```Pipeline``` object then provides methods for defining a sequence of processing steps to apply to each record. These processing steps include:

- Passing user-defined functions to the pipeline via the ```map``` method.

or...

- Using a set of built-in methods, such as ```fetch```, ```fetch_dataset```, ```align```, ```keep```, or ```discard```.

The ```Pipeline``` also provides a ```where``` method which takes as input a user-defined function that returns a boolean value. If the function evaluates to ```False``` for a record, then that record is removed from the pipeline.


## Talk to your data

TokSearch ships with `toksearch.llm` — a conversational interface that lets you ask for fusion data in plain English. The agent writes the pipeline code, runs it against a persistent Python namespace so follow-up turns iterate on cached results instead of re-fetching, and shows you each block before executing it. From a shell:

```bash
toksearch chat
```

```text
you> Use run_python to fetch ipmhd for shot 165920 from efit01.
[run_python] Fetch ipmhd via MdsSignal.
  pipeline = toksearch.Pipeline([165920])
  pipeline.fetch('ip', toksearch.MdsSignal(r'\ipmhd', 'efit01'))
  rec = list(pipeline.compute_serial())[0]
[output] (no output)

you> What's the peak |Ip| in MA?
[run_python] ...
[output] 1.1325
```

Backends: Anthropic API, OpenAI API, your Claude Max plan via the Claude Agent SDK, or GA's AmSC endpoint via `toksearch_d3d`. See the [LLM tutorial](https://ga-fdp.github.io/toksearch/LLM_Tutorial/) for an end-to-end walkthrough and the [LLM Interface reference](https://ga-fdp.github.io/toksearch/llm/) for the full API surface.


## Installation

TokSearch is available on the `ga-fdp` conda channel. 

In the near future, we will provide a way to install TokSearch directly from PyPI using pip.


### Installation with Conda in an existing environment

To install in an existing conda environment, run:



```bash
conda install -c ga-fdp -c conda-forge toksearch
```

or equivalently

```bash
conda install -c conda-forge ga-fdp::toksearch
```

You can substitute `mamba` for `conda` if you prefer.

### Installation with Conda in a new environment
Optionally, you can create a new environment:

```bash
mamba create -n toksearch -c ga-fdp -c conda-forge toksearch
```

### Installation from Source

At the moment, the cleanest way to install TokSearch from source is to first set up a Conda/Mamba environment with the required dependencies, and then install TokSearch from the local clone of the repository. Here are the steps:

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

