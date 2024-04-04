
# GA TokSearch

TokSearch is a Python package for parallel retrieving, processing, and filtering of arbitrary-dimension fusion experimental data. TokSearch provides a high level API for extracting information from many shots, along with useful classes for low level data retrieval and manipulation.

The fundamental class in TokSearch is the ```Pipeline```. A ```Pipeline``` object takes a list of shots and, for each shot in the list, creates a dict-like object called a ```Record```. The ```Pipeline``` object then provides methods for defining a sequence of processing steps to apply to each record. These processing steps include:

- Passing user-defined functions to the pipeline via the ```map``` method.

or...

- Using a set of built-in methods, such as ```fetch```, ```fetch_dataset```, ```align```, ```keep```, or ```discard```.

The ```Pipeline``` also provides a ```where``` method which takes as input a user-defined function that returns a boolean value. If the function evaluates to ```False``` for a record, then that record is removed from the pipeline. 




# Basic tutorial

The following example illustrates a simple processing pipeline and demonstrates the key parts of the API. 

The basic workflow with TokSearch is to first define a set of ```Signal``` objects which represent, for example, MDSplus or PTDATA pointnames. Here we grab the measured plasma current, ```ip```, from PTDATA and the calculated plasma current, ```ipmhd```, from the efit01 MDSplus tree.


```python
from toksearch import PtDataSignal, MdsSignal

ip_signal = PtDataSignal('ip')
ipmhd_signal = MdsSignal(r'\ipmhd', 'efit01', location='remote://atlas.gat.com')
```

Next, we instantiate a ```Pipeline``` object with a list of the shots that we want to process.


```python
from toksearch import Pipeline

shots = [165920, 165921]
pipeline = Pipeline(shots)
```

We then pass the ```pipeline``` object the previously created signal objects and give them labels using the ```fetch``` method. A key point to understand here is that the ```fetch``` method does not immediately retrieve the requested data. It defers execution until explicitly requested (more on this later).


```python
pipeline.fetch('ip', ip_signal)
pipeline.fetch('ipmhd', ipmhd_signal)
```

At this point we can inspect what the pipeline is doing by running one of the ```compute*``` family of methods. At the time of this writing, there are three supported ways of running the pipeline: ```compute_serial```, ```compute_spark```, and ```compute_ray```.

```compute_serial```, as the name suggests, processes each shot serially. In our current example, it would process shot 165920, followed by 165921.

The other methods will partition the list of shots into roughly equal sized chunks and process those chunks in parallel using the specified distributed computing framework (i.e. Apache Spark or Ray).

For our example we'll use ```compute_serial```. All of these method return a list-like object that contains the resulting sequence of records.


```python
import numpy as np
import collections
import pprint


np.set_printoptions(threshold=3, precision=1)

records = pipeline.compute_serial()
print('Number of records: {}. Should be 2.'.format(len(records)))

# Helper function for printing results
def pretty_print(record):
    print('*'*80)
    for key in record.keys():
        val = record[key]
        if isinstance(val, collections.Mapping):
            print('{}:'.format(key))
            for subkey, subval in val.items():
                print('\t{}: {}'.format(subkey, subval))
        else:
           print('{}: {}'.format(key, val))

# Note the list-like behavior of the records result
for record in records:
    pretty_print(record)
 
```

    Number of records: 2. Should be 2.
    ********************************************************************************
    shot: 165920
    errors:
    ip:
    	data: [-1391.2  -463.7   154.6 ...  -463.7 -1391.2 -2318.7]
    	times: [ -996.   -995.5  -995.  ... 14362.5 14363.  14363.5]
    ipmhd:
    	data: [213493.6 281801.6 286739.6 ... 475302.8 474772.5 471811.7]
    	times: [ 100.  140.  160. ... 6340. 6360. 6380.]
    ********************************************************************************
    shot: 165921
    errors:
    ip:
    	data: [  1391.2      0.  -10511.6 ...  -2164.2  -1236.7   -927.5]
    	times: [ -996.   -995.5  -995.  ... 14362.5 14363.  14363.5]
    ipmhd:
    	data: [213254.  282727.8 286100.  ... 485212.5 483553.4 481211.4]
    	times: [ 100.  140.  160. ... 6340. 6360. 6380.]


Some things to note about the result:

- Each ```Record``` object in ```results``` will always contain two attributes: ```shot``` and ```errors```. The ```errors``` attribute is a dictionary that stores information about any exceptions that occurred during execution of the pipeline for that shot. In this case, there were no exceptions, so the ```errors``` attribute is just an empty dict.

- Recall that when we made the calls to ```pipeline.fetch(...)```, we specified the labels 'ip' and 'ipmhd'. Those now show up in the results records as fields. The default behavior for the ```fetch``` operation is to return a dictionary with the fields ```data``` and ```times```, each of which is a numpy array.

At this point we haven't done anything terribly interesting. Let's now do some more processing by applying a ```map``` operation to each ```Record``` in the pipeline.

We'll define a function, ```max_currents```, that calculates the maximum absolute value of both ```ip``` and ```ipmhd```. Functions passed to ```map``` take a single ```Record``` object as input, and then modify that object in place (returning nothing).


```python
@pipeline.map
def max_currents(record):
    record['max_ip'] = np.max(np.abs(record['ip']['data']))
    record['max_ipmhd'] = np.max(np.abs(record['ipmhd']['data']))
```

Note that we're using the decorator formulation of ```max_currents```. We could equivalently have done this:

```python
def max_currents(record):
    record['max_ip'] = np.max(np.abs(record['ip']['data']))
    record['max_ipmhd'] = np.max(np.abs(record['ipmhd']['data']))
    
pipeline.map(max_currents)
```

Let's run ```compute_serial``` again and examine the results:


```python
records = pipeline.compute_serial()

for record in records:
    pretty_print(record)
```

    ********************************************************************************
    shot: 165920
    errors:
    ip:
    	data: [-1391.2  -463.7   154.6 ...  -463.7 -1391.2 -2318.7]
    	times: [ -996.   -995.5  -995.  ... 14362.5 14363.  14363.5]
    ipmhd:
    	data: [213493.6 281801.6 286739.6 ... 475302.8 474772.5 471811.7]
    	times: [ 100.  140.  160. ... 6340. 6360. 6380.]
    max_ip: 1152410.4080200195
    max_ipmhd: 1129914.0
    ********************************************************************************
    shot: 165921
    errors:
    ip:
    	data: [  1391.2      0.  -10511.6 ...  -2164.2  -1236.7   -927.5]
    	times: [ -996.   -995.5  -995.  ... 14362.5 14363.  14363.5]
    ipmhd:
    	data: [213254.  282727.8 286100.  ... 485212.5 483553.4 481211.4]
    	times: [ 100.  140.  160. ... 6340. 6360. 6380.]
    max_ip: 1143908.3862304688
    max_ipmhd: 1124894.625


The two records now have the fields ```max_ip``` and ```max_ipmhd``` as expected.

For this simple example we are gathering all of the raw data used to calculate ```max_ip``` and ```max_ipmhd```. But, for cases with many more shots or many more pointnames, we can easily exceed the memory on the local machine. In those cases it's wise to only return the calculated quantities that we care about. We can use the ```Pipeline``` methods ```keep``` or ```discard``` to achieve this.


```python
pipeline.keep(['max_ip', 'max_ipmhd'])

records = pipeline.compute_serial()
for record in records:
    pretty_print(record)
```

    ********************************************************************************
    shot: 165920
    errors:
    max_ip: 1152410.4080200195
    max_ipmhd: 1129914.0
    ********************************************************************************
    shot: 165921
    errors:
    max_ip: 1143908.3862304688
    max_ipmhd: 1124894.625


Note that the ```ip``` and ```ipmhd``` fields are no longer present in the records.

Now let's suppose that we want to only find shots for which the maximum ```ip``` is above 1.15 MA. We implement this condition by creating a user-defined function that returns a boolean value. When run by the pipeline, if this function returns ```False``` the record will be removed from the pipeline.


```python
@pipeline.where
def max_ip_is_high_enough(record):
    return record['max_ip'] > 1.15e6

records = pipeline.compute_serial()
print('len(records): {}. Should be 1.'.format(len(records)))


for record in records:
    pretty_print(record)
```

    len(records): 1. Should be 1.
    ********************************************************************************
    shot: 165920
    errors:
    max_ip: 1152410.4080200195
    max_ipmhd: 1129914.0


Only one of the two input shots (165920) matched the where criteria, so the length of ```records``` is 1.

# Where to go next

- [Parallelization](Parallelization.ipynb)
- [Using TokSearch with Xarray](Using%20with%20Xarray.ipynb)
- [Working with Signals](Working%20with%20Signals.ipynb)

