import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Iterable, List
import tqdm
import logging
import time
import cProfile, pstats

from datagossip.datagossip import DataGossipLoader, DataGossipLoss
from datagossip.dataset import load_dataset, DistributedDataLoader
from datagossip.instance_selector import InstanceSelectorChooser
from datagossip.models import ModelSize
from datagossip.models.early_stopping import EarlyStopping
from datagossip.optim import DownpourSGD, DownpourAdagrad
from datagossip.server import ParameterServer
from datagossip.utils.distributed import Cluster
from datagossip.utils.experiments import Experiment


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def ownBool(x: str) -> bool:
    return x == "True"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, help="Cluster rank")
    parser.add_argument('--size', type=int, help="Cluster size")
    parser.add_argument('--master_address', type=str, default="localhost", help="Cluster master address")
    parser.add_argument('--master_port', type=int, default=29900, help="Cluster master port")
    parser.add_argument('--datagossip', type=ownBool, default=True, help="DataGossip activated?")
    parser.add_argument('--optimizer', type=str, default="sgd", help="Which optimizer")
    parser.add_argument('--instance_selector', type=InstanceSelectorChooser, choices=InstanceSelectorChooser, default="active_bias", help="Which instance selector for DataGossip")
    parser.add_argument('--early_stopping', type=ownBool, default=False, help="Early Stopping activated?")
    parser.add_argument('--patience', type=int, default=10, help="Early stopping after {patience} epochs")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=0.003, help="Initial learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size for SGD")
    parser.add_argument('--n_push_pull', type=int, default=5, help="Update model every n iterations")
    parser.add_argument('--n_gather', type=int, default=4, help="Update selected instances n times per epoch")
    parser.add_argument('--k', type=int, default=50, help="Number of points being gossiped")
    parser.add_argument('--overlap', type=int, default=0, help="Overlap for baseline")
    parser.add_argument('--dataset', type=str, default='mnist', help="Dataset used for training")
    parser.add_argument('--model', type=ModelSize, choices=ModelSize, default='large', help="Model used for training")
    parser.add_argument('--imbalanced', type=ownBool, default=True, help="Dataset partitioning imbalanced?")
    parser.add_argument('--local_tests', type=ownBool, default=False)
    parser.add_argument('--slowout', type=int, default=0, help="Number of nodes running with low priority processes")
    parser.add_argument('--remote_train_frequency', type=int, default=1, help="After how many local training steps should a remote training follow?")
    parser.add_argument('--parameter_server', type=ownBool, default=True, help="Use a Parameter Server")
    args = parser.parse_args(sys.argv[1:])

    print(args)

    (dataset, test_dataset), (data_loader, test_loader) = distribute_datasets(args)
    model = args.model.get_model_by_size(args.dataset)

    if args.datagossip:
        data_loader, criterion = prepare_datagossip(data_loader, args)
        rank = args.rank * 2
        size = args.size * 2
        local_world_size = 2
    else:
        criterion = nn.NLLLoss()
        rank = args.rank
        size = args.size
        local_world_size = 1

    with Cluster(rank, size, args.master_address, args.master_port):
        sgd_ranks = [r for r in range(dist.get_world_size()) if (r % local_world_size) == 0]
        sgd_group = dist.new_group(ranks=sgd_ranks)

        if args.datagossip:
            dist.new_group(ranks=[r for r in range(2, dist.get_world_size()) if (r % local_world_size) == 1])

        if dist.get_rank() == 0:
            if args.datagossip:
                data_loader.stop()
            if args.parameter_server:
                parameter_server(model, group=sgd_group, client_ranks=sgd_ranks[1:], args=args, test_loader=test_loader)
        else:
            train(
                model,
                data_loader,
                test_loader,
                criterion,
                sgd_group,
                args
            )


def distribute_datasets(args) -> Tuple[Tuple[TensorDataset, TensorDataset], Tuple[DataLoader, DataLoader]]:
    dataset, test_dataset = load_dataset(args.dataset)

    with Cluster(args.rank, args.size, args.master_address, args.master_port):
        data_loader = DistributedDataLoader(dataset,
                                            partition=True,
                                            parameter_server=True,
                                            batch_size=args.batch_size,
                                            imbalanced=args.imbalanced,
                                            shuffle=True,
                                            # minus parameter server and own node
                                            overlap=args.overlap,
                                            with_indices=args.datagossip)
        test_loader = DistributedDataLoader(test_dataset,
                                            partition=False,
                                            parameter_server=True,
                                            batch_size=256)

    return (dataset, test_dataset), (data_loader, test_loader)


def prepare_datagossip(data_loader: DataLoader, args) -> Tuple[Iterable, nn.NLLLoss]:
    dg_loader = DataGossipLoader(data_loader,
                                 instance_selector=args.instance_selector,
                                 data_shape=data_loader.dataset.tensors[0].shape[1:],
                                 args=args)

    if args.model == "large":
        loss_fn = nn.functional.nll_loss
    else:
        loss_fn = nn.functional.cross_entropy

    dg_loss = DataGossipLoss(dg_loader.instance_selector, loss_fn=loss_fn)

    return dg_loader, dg_loss


def parameter_server(model: nn.Module, group: dist.group, client_ranks: List[int], args, test_loader: DataLoader):
    logger.debug("parameter server started")
    server = ParameterServer(model=model if args.model == "large" else model.classifier, group=group, client_ranks=client_ranks, args=args, test_loader=test_loader, test_model=model)
    server.start()
    logger.debug("parameter server stopped")


def resize_data(data: torch.Tensor, args, size: int = 224):
    if args.model != "large":
        data = F.interpolate(data, size=size)
    return data


@torch.no_grad()
def test(model: nn.Module, data_loader: DataLoader, args):
    model.eval()
    correct = 0
    for data, target in data_loader:
        data = resize_data(data, args)
        output = model(data)
        pred = output.max(1)[1]
        correct += pred.eq(target).sum().item()
    acc = correct / len(data_loader.dataset)
    model.train()
    return acc


def train(model: nn.Module, data_loader: DataLoader, test_loader: DataLoader, criterion: nn.NLLLoss, group: dist.group, args):
    early_stopping = EarlyStopping(patience=args.patience) if args.early_stopping else None

    if args.optimizer == "sgd":
        optim_class = DownpourSGD
    elif args.optimizer == "adagrad":
        optim_class = DownpourAdagrad
    else:
        raise ValueError(f"Please choose either 'sgd' or 'adagrad' as optimizer! Wrong input: '{args.optimizer}'")

    parameters = model.parameters() if args.model == "large" else model.classifier.parameters()

    optimizer = optim_class(parameters,
                            lr=args.lr,
                            n_pull=args.n_push_pull,
                            n_push=args.n_push_pull,
                            model=model if args.model == "large" else model.classifier,
                            group=group)

    with Experiment(".", metrics=["acc", "process_time"], attributes=dict(args._get_kwargs())) as experiment:
        for e in range(args.epochs):
            for data, target in tqdm.tqdm(data_loader, desc=f"Epoch {e + 1}"):
                data = resize_data(data, args)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            if args.local_tests:
                acc = test(model, test_loader, args)
                logger.warning(f"test acc: {acc}")
                experiment.add_results(e, acc, time.process_time())
                if early_stopping is not None and early_stopping(acc):
                    logger.debug(f"early stopping after {early_stopping.patience} epochs of patience")
                    break

    optimizer.kill_master()
    if args.datagossip:
        data_loader.stop()


if __name__ == "__main__":
    if "--profiler=on" in sys.argv:
        pr = cProfile.Profile()
        pr.enable()
        main()
        pr.disable()
        with open(f"stats_{time.time()}.txt", "w") as f:
            ps = pstats.Stats(pr, stream=f).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats()
    else:
        main()
