import torch
from typing import List, Any
from torch.utils.data import DataLoader
from ...utils.data_and_sampling_selector import setNegativeSamplingAlgorithm
from ...negative_sampling.hypergraph_negative_sampling_algorithm import HypergraphNegativeSampler, MotifHypergraphNegativeSampler
from torch_geometric.data.hypergraph_data import HyperGraphData
from ..datasets.dataset_hypergraph import DatasetHyperGraph


class DatasetLoader(DataLoader):
    """ A class data loader which merge data object from a dataset 
        to a mini-batch.
        
        Args:
            dataset (DatasetHyperGraph): The dataset from which to load the data.
            batch_size (int, optional): How many samples per batch to load.
                (default: 1)
            shuffle (bool, optional): Set True to have data reshuffled at every epoch.
                (default: False)
            **kwargs: Additional arguments for the class.
    """

    def __init__(self, dataset: DatasetHyperGraph, negative_sampling: str, alpha: float, beta: int, num_node: int, batch_size: int = 1, shuffle: bool = False, **kwargs):
        kwargs.pop("collate_fn", None)
        
        hypergraph_negative = setNegativeSamplingAlgorithm(negative_sampling, num_node, alpha, beta).generate(dataset._data.edge_index)
        dataset.edge_index = hypergraph_negative.edge_index

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn= self.collate,
            **kwargs
        )
    
    def collate(self, batch: List[Any]) -> HyperGraphData:
        x = batch[0].x
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0, batch[0].edge_attr.shape[1]), dtype= torch.long)
        num_nodes = 0

        for i in range(len(batch)):
            b = batch[i]
            num_nodes += b.num_nodes
            b_edge_index = b.edge_index
            b_edge_index[1] += i
            edge_index = torch.hstack((edge_index, b_edge_index))
            b_edge_attr = b.edge_attr
            edge_attr = torch.vstack((edge_attr, b_edge_attr))
        unique, edge_index[0] = edge_index[0].unique(return_inverse=True)
        return  HyperGraphData(
            x = x[unique],
            edge_index= edge_index,
            edge_attr= edge_attr,
            num_nodes= num_nodes,
            num_edges=edge_index[1].max().item() + 1,
        )
        
class DatasetLoaderIMDB(DataLoader):
    def __init__(self, dataset: DatasetHyperGraph, batch_size: int = 1, shuffle: bool = False, **kwargs):
        kwargs.pop("collate_fn", None)

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn= self.collate_fn,
            **kwargs
        )
        
    def collate_fn(self, batch):
        x = batch[0].x
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, batch[0].edge_attr.shape[1]), dtype=torch.long)
        for i in range(len(batch)):
            b = batch[i]
            b_edge_index = b.edge_index
            b_edge_index[1] += i
            edge_index = torch.hstack((edge_index, b_edge_index))
            b_edge_attr = b.edge_attr
            edge_attr = torch.vstack((edge_attr, b_edge_attr))
        unique, edge_index[0] = edge_index[0].unique(return_inverse=True)
        result = HyperGraphData(
            x=x[unique],
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        return result
