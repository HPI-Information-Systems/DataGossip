# DataGossip

DataGossip [[paper](http://dx.doi.org/10.48786/edbt.2022.24)] is an extension for asynchronous distributed data parallel machine learning that improves the training on imbalanced partitions.

## Installation

requires conda:

```shell script
$ conda env create -f environment.yml
$ conda activate datagossip
$ python setup.py install
```

## Experiment Reproducibility
Download and transform the datasets on your main machine:

```shell script
$ python prepare_datasets.py
```

Then, run the following script on each cluster node to start the training. Be aware to set the right __ranks__ and __sizes__!

```shell script
$ python experiments/train.py --rank=<rank> --size=<size> --main_address=<main_address> 
```

Afterwards, you can find the results of the experiment in the files (on your machine with rank=0) _experiments.pkl_ and _evaluations.pkl_ which hold pandas DataFrames.

## Reference

Please consider citing:
```bibtex
@inproceedings{wenig2022datagossip,
  title={DataGossip: A Data Exchange Extension for Distributed Machine Learning Algorithms},
  author={Wenig, Phillip and Papenbrock, Thorsten},
  booktitle={Proceedings of the International Conference on Extending Database Technology (EDBT)},
  year={2022},
  pages={373--377},
  doi={10.48786/edbt.2022.24},
  url={http://dx.doi.org/10.48786/edbt.2022.24},
}
```
