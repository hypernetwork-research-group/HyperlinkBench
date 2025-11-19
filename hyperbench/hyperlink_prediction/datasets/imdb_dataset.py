import torch
import pickle
from abc import ABC
from .dataset_hypergraph import DatasetHyperGraph
from torch_geometric.data.hypergraph_data import HyperGraphData


class CHLPBaseDataset(DatasetHyperGraph, ABC):
    """A object that save a hypergraph dataset from Google Drive and load in memory

        Args:
            dataset_name (string): The dataset's name.
            root (string): The root where the dataset will be saved.
                 (default: 'datasets')
    """
    GDRIVE_ID = None
    DATASET_NAME = None

    def __init__(self,
                 root: str = 'data',
                 *args, **kwargs):
        super().__init__(dataset_name= self.DATASET_NAME ,root= root, *args, **kwargs)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return f"{self.root}/{self.DATASET_NAME}/raw"

    @property
    def processed_dir(self):
        return f"{self.root}/{self.DATASET_NAME}/processed"

    @property
    def raw_file_names(self):
        return ["hyperedges.txt", "node_features.txt", "hyperedge_features.txt", "hyperedge_embeddings.txt", "node_embeddings.txt"]

    @property
    def processed_file_names(self):
        return "processed.pt"

    def download(self):
        """ Take the dataset from Google Drive through the name of dataset,
            which is associeted the key, and download it in memory in the folder
            datasets.
        """
        from os import listdir
        if len(listdir(self.raw_dir)) > 0:
            return
        from gdown import download
        archive_file_name = self.raw_dir + "/" + "raw.zip"
        download(id=self.GDRIVE_ID, output=archive_file_name)
        import zipfile
        with zipfile.ZipFile(archive_file_name, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        from os import remove
        remove(archive_file_name)

    def process(self):
        """ Process the files of the verts and the simplices in a tensor containing two list
            the first contain the nodes's id
            and the second the index of the nodes's numbers
            and then serialize the tensor 
        """
        edge_index = [[], []]
        with open(self.raw_dir + "/hyperedges.txt", "r") as f:
            for i, line in enumerate(f):
                for l in line.split():
                    edge_index[0].append(int(l))
                    edge_index[1].append(i)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        with open(self.raw_dir + "/node_embeddings.pkl", "rb") as f:
            node_embeddings = torch.tensor(pickle.load(f))
        with open(self.raw_dir + "/hyperedge_embeddings.pkl", "rb") as f:
            hyperedge_embeddings = torch.tensor(pickle.load(f))

        data_list = [HyperGraphData(
            x=node_embeddings,
            edge_index=edge_index,
            edge_attr=hyperedge_embeddings,
        )]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def __getitem__(self, idx) -> HyperGraphData:
        if isinstance(idx, int):
            idx = torch.tensor([idx])
        if isinstance(idx, list):
            idx = torch.tensor(idx)
        edge_index_mask = torch.isin(self._data.edge_index[1], idx)
        edge_index = self._data.edge_index[:, edge_index_mask]
        _, edge_index[1] = edge_index[1].unique(return_inverse=True)
        return HyperGraphData(
            x=self._data.x,
            edge_index=edge_index,
            edge_attr=self._data.edge_attr[idx],
        )

    def __len__(self) -> int:
        return self._data.edge_index[1].max().item() + 1

class IMDBHypergraphDataset(CHLPBaseDataset):
    
    GDRIVE_ID = "1D-dqEmkOfOVy6w0ZfrLtJrPw-dibwJ3V"
    DATASET_NAME = "IMDB"

class ARXIVHypergraphDataset(CHLPBaseDataset):
    
    GDRIVE_ID = "1pXRgFzVKIC-WtSslapAEPf9KlerTFhnd"
    DATASET_NAME = "ARXIV"

class COURSERAHypergraphDataset(CHLPBaseDataset):
    
    GDRIVE_ID = "1AVqUuVOpCFVG13-N3lUckYhIrk83WOSu"
    DATASET_NAME = "COURSERA"

torch.serialization.add_safe_globals([HyperGraphData])
