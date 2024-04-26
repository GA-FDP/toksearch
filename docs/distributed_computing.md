# Distributed computing with TokSearch

The mechanism for distributing TokSearch computations is to invoke a TokSearch script using the ```toksearch_submit``` utility. Behind the scenes, ```toksearch_submit``` uses SLURM, and can be run in both interactive and batch modes, utilizing salloc and sbatch respectively. The default mode is to run interactively, using salloc.

When using the `Pipeline.compute_ray()` or `Pipeline.compute_spark()` methods, the computation will be automatically distributed across the nodes allocated by `toksearch_submit`.

## How TokSearch uses Ray and Spark under SLURM

When a TokSearch script is run with `toksearch_submit` using N nodes, the script will be executed on one of the worker nodes, and the Ray or Spark cluster will be started on the allocated nodes, with the master node running on the same node as the script, and the worker nodes running on the other N-1 nodes.


The base SLURM commands that toksearch_submit runs are of the following form:

For interactive jobs:

```bash
 salloc -N NUM_NODES --exclusive ${interactive.salloc} \
   srun -u --pty -N 1 --exclusive --propagate ${interactive.srun} \
   COMMAND_TO_RUN ARGS
```

 For batch jobs:

```bash
 sbatch -N NUM_NODES --exclusive ${batch.sbatch} \
   COMMAND_TO_RUN ARGS
```

 Options and arguments in CAPS are provided by the user when running
 toksearch_submit. The options and arguments in braces 
 (e.g. `${batch.sbatch}`) are injected from the configuration file (see below).


## Configuration file

`toksearch_submit` uses YAML a configuration file to specify memory and CPU requirements for the Ray and Spark backends, and to adapt to the site-specific SLURM configuration.

Below, we walk through the different sections of the default config file. The actual file is commented with more detailed information, mirroring the documetation here.

```yaml
# Job options passed through to salloc/sbatch

# Options for batch jobs (ie those using sbatch)
batch:
    sbatch:
      - --time=4:00:00

# Options for interactive jobs (ie those using salloc)
interactive:
    salloc:
      - --time=4:00:00
      - --x11=all

    srun:
      - --overlap
```
Spark processes are started using the srun command (when compute_spark
 is invoked on a Pipeline object inside a SLURM job).

 The master process is started with the following command, with sections
 in braces replaced by the configuration options, and options with 
 underscore + CAPS (e.g. _PORT) filled automatically by toksearch_submit:

```bash 
 srun --nodes=1 --ntasks=1 -w _HOST ${spark.master.srun}  \
   spark-class org.apache.spark.deploy.master.Master \
     --host _HOST \
     --port _PORT \
     ${spark.master.start}
```

 Similarly, the worker process is started with the following command:

```bash
 srun --nodes=1 --ntasks=1 -w _HOST ${spark.worker.srun}  \
   spark-class org.apache.spark.deploy.worker.Worker \
     -i _HOST \
     ${spark.worker.start} \
     _MASTER_URL
```

```yaml
spark:
    # Options for the Spark master
    master:
        srun:
            - --overlap

    worker:
        srun:
            - --overlap
        start:
            - -c
            - 10
            - -m
            - 24G
```


 Ray processes are started using the srun command (when compute_ray
 is invoked on a Pipeline object inside a SLURM job).

 The master process is started with the following command, with sections
 in braces replaced by the configuration options, and options with 
 underscore + CAPS (e.g. _PORT) filled automatically by toksearch_submit:

```bash
srun --nodes=1 --ntasks=1 -w _HOST ${ray.master.srun}  \
  ray start 
  --head \
  --node-ip-address _IP \
  --port _PORT \
  --temp-dir _TEMP_DIR \
   ${ray.master.start} 
```
 
 Similarly, the worker process is started with the following command:

```bash 
 srun --nodes=1 --ntasks=1 -w _HOST ${ray.worker.srun}  \
  ray start
  --node-ip-address _IP \
  --address _MASTER_ADDRESS:_MASTER_PORT \
  --temp-dir _TEMP_DIR \
  ${ray.worker.start}
```

```yaml
ray:
    master:
        srun:
            - --overlap
            - --propagate

        # Ray memory options can be adjusted here
        #start:
          #- --object-store-memory=95000000000
          #- --memory=80000000000

    worker:
        srun:
            - --overlap
            - --propagate

        # Ray memory options can be adjusted here
        #start:
          #- --object-store-memory=95000000000
          #- --memory=80000000000
```

## Example script

This will be demonstrated with an example script:

```python
# toksearch_example.py
import argparse
from toksearch import Pipeline
from toksearch import MdsSignal


def create_pipeline(shots):
    ipmhd_signal = MdsSignal(r"\ipmhd", "efit01")

    pipeline = Pipeline(shots)
    pipeline.fetch("ipmhd", ipmhd_signal)

    @pipeline.map
    def calc_max_ipmhd(rec):
        rec["max_ipmhd"] = np.max(np.abs(rec["ipmhd"]["data"]))

    pipeline.keep(["max_ipmhd"])
    return pipeline


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["spark", "ray"])
    args = parser.parse_args()

    backend = args.backend

    num_shots = 10000
    shots = list(range(165920, 165920 + num_shots))

    pipeline = create_pipeline(shots)

    if backend == "ray":
        results = pipeline.compute_ray()
    else:  # spark
        results = pipeline.compute_spark()

    print(f"Got {len(results)} results using {backend}")
```

This script will fetch the IPMHD signal from the MDSplus tree `efit01` for a range of shots, and calculate the maximum value of the signal. The script will then print the number of results and the backend used.

## Running the script with `toksearch_submit`

First, we'll run with the Ray backend:

```bash
toksearch_submit -N 3 python toksearch_example.py -- ray
```

with the following output:

```
salloc -N 3 --exclusive --gres=gpu:volta:1 --x11=all srun -u --pty -N 1 --exclusive --propagate --gres=gpu:volta:0 --overlap python toksearch_example.py ray
salloc: Granted job allocation 16902
salloc: Waiting for resource configuration
salloc: Nodes saga[03-05] are ready for job
Initializing ray with _temp_dir = /tmp/tmp1ei083q3
['10.0.0.43', '10.0.0.44', '10.0.0.45']
STARTING CLUSTER
--nodes=1 --ntasks=1 -w saga03 --gres=gpu:volta:1 --overlap --propagate ray start --block --node-ip-address 10.0.0.43 --temp-dir /tmp/tmp1ei083q3 --port 6543 --head --object-store-memory=95000000000 --memory=80000000000
Ok, started head node
dict_keys(['saga04', 'saga05'])
Starting saga04...
--nodes=1 --ntasks=1 -w saga04 --gres=gpu:volta:1 --overlap --propagate ray start --block --node-ip-address 10.0.0.44 --temp-dir /tmp/tmp1ei083q3 --address=10.0.0.43:6543 --object-store-memory=95000000000 --memory=80000000000
Starting saga05...
--nodes=1 --ntasks=1 -w saga05 --gres=gpu:volta:1 --overlap --propagate ray start --block --node-ip-address 10.0.0.45 --temp-dir /tmp/tmp1ei083q3 --address=10.0.0.43:6543 --object-store-memory=95000000000 --memory=80000000000
********************************************************************************
BATCH 1/1
NUM CPUS: 144
NUM PARTITIONS: 10000
MEDIAN PARTITION SIZE: 1
Got 10000 results using ray
salloc: Relinquishing job allocation 16902
```


Next, we'll run with the Spark backend:

```bash
toksearch_submit -N 3 python toksearch_example.py -- spark
```

with the following output:

```
salloc -N 3 --exclusive --gres=gpu:volta:1 --x11=all srun -u --pty -N 1 --exclusive --propagate --gres=gpu:volta:0 --overlap python toksearch_example.py spark
salloc: Granted job allocation 16908
salloc: Waiting for resource configuration
salloc: Nodes saga[04-06] are ready for job
['10.0.0.44', '10.0.0.45', '10.0.0.46']
STARTING CLUSTER
MASTER IP 10.0.0.44
['--nodes=1', '--ntasks=1', '-w', 'saga04', '--gres=gpu:volta:0', '--overlap', 'spark-class', 'org.apache.spark.deploy.master.Master', '--host', '10.0.0.44', '--port', 7077]
Ok, started head node
dict_keys(['saga04', 'saga05', 'saga06'])
Starting saga04...
['--nodes=1', '--ntasks=1', '-w', 'saga04', '--gres=gpu:volta:1', '--overlap', 'spark-class', 'org.apache.spark.deploy.worker.Worker', '-i', '10.0.0.44', '-c', 48, '-m', '149G', 'spark://10.0.0.44:7077']
Starting saga05...
['--nodes=1', '--ntasks=1', '-w', 'saga05', '--gres=gpu:volta:1', '--overlap', 'spark-class', 'org.apache.spark.deploy.worker.Worker', '-i', '10.0.0.45', '-c', 48, '-m', '149G', 'spark://10.0.0.44:7077']
Starting saga06...
['--nodes=1', '--ntasks=1', '-w', 'saga06', '--gres=gpu:volta:1', '--overlap', 'spark-class', 'org.apache.spark.deploy.worker.Worker', '-i', '10.0.0.46', '-c', 48, '-m', '149G', 'spark://10.0.0.44:7077']
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
24/04/15 09:51:48 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Got 10000 results using spark                                                   
salloc: Relinquishing job allocation 16908
```


