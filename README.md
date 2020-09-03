# DataGossip

DataGossip is an extension for asynchronous distributed data parallel machine learning that improves the training on imbalanced partitions.

## Installation

requires conda:

```shell script
$ conda env create -f environment.yml
$ conda activate datagossip
$ python setup.py install
```

## Experiment Reproducibility
Download and transform the datasets on your master machine:

```shell script
$ python prepare_datasets.py
```

Then, run the following script on each cluster node to start the training. Be aware to set the right __ranks__ and __sizes__!

```shell script
$ python experiments/train.py --rank=<rank> --size=<size> --master_address=<master_address> 
```

Afterwards, you can find the results of the experiment in the files (on your machine with rank=0) _experiments.pkl_ and _evaluations.pkl_ which hold pandas DataFrames.
