import torch
import torch_geometric.nn.aggr as aggr
from enum import Enum
from torch import Tensor
from .hypergraph_negative_sampling import HypergraphNegativeSampler
from .hypergraph_negative_sampling_result import HypergraphNegativeSamplerResult, ABSizedHypergraphNegativeSamplerResult

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class ABSizedHypergraphNegativeSampler(HypergraphNegativeSampler):
    """ A class Negative Sampler which samples the negative sample using
        the alpha and beta algorithm integreted whit the Size Negative Sampling.

        Args:
            num_node (int): The hypergraph's number of nodes.
            alpha (float | int): A costant which indicate the genuiness of the negative hyperlink.
                (default: 0.5)
            beta (int): A constant which indicate the ratio between positive and negative hyperlink.
                (default: 1)
            mode (Mode): Indicate how to calculate the probabilities of the negative sample.
                (default: Mode.BEST_EFFORT)
            avoid_duplicate_node (bool): A value which avoid the duplicate node.
                (default: True)
    """

    class Mode(Enum):
        BEST_EFFORT = "best"
        NODE_AWARE = "node"
        HYPEREDGE_AWARE = "hyperedge"

    def __init__(self, num_node, alpha: float | int = 0.5, beta: int = 1, mode: Mode = Mode.BEST_EFFORT, avoid_duplicate_nodes: bool = True):
        super().__init__(num_node)
        self.alpha = alpha
        if self.alpha >= 1 and self.alpha != int(self.alpha):
            raise ValueError("If alpha is greater than or equal to 1, it must be an integer")
        self.beta = beta
        self.mode = mode
        self.avoide_duplicate_nodes = avoid_duplicate_nodes

    def fit(self, *args):
        return self
    
    #Parameter edge_index unused
    def get_probabilities(self, replace_mask: torch.Tensor, edge_index: torch.Tensor) -> float:
        probabilities = torch.ones((replace_mask.sum().item(), self.num_node), device = self.device)
        probabilities /= probabilities.sum(dim = 1, keepdim = True)
        return probabilities
    
    def get_replace_mask(self, edge_index: torch.Tensor) -> Tensor:
        if self.alpha >= 1:
            replace_mask = torch.zeros(edge_index.shape[1], dtype = torch.bool, device = self.device)
            """ The degrees return a tensor which for every position return the number of the verts
                presents in a hyperedge, like the file with the exention nverts.txt
            """
            degrees = aggr.SumAggregation()(
                torch.ones((edge_index.shape[1],1), dtype = torch.float32, device = self.device),
                edge_index[1]
            ).flatten().long()
            cursor = 0 
            for e in torch.unique(edge_index[1]):
                if degrees[e] <= self.alpha:
                    replace_mask[cursor:cursor + degrees[e]] = True
                else:
                    choise = torch.randint(0, degrees[e], (self.alpha,), device= self.device)
                    replace_mask[cursor + choise] = True
                cursor += degrees[e]
        else:
            replace_mask = torch.rand(edge_index.shape[1], device=self.device) >= self.alpha
            while True: #Ensure that at least one node is replaced in all hyperedges
                unchanged = (aggr.SumAggregation()(
                    replace_mask.float().view(-1,1),
                    edge_index[1]
                ) == 0).flatten().bool()
                if not unchanged[torch.unique(edge_index[1])].any():
                    break
                unchanged = torch.isin(edge_index[1], unchanged.nonzero())
                replace_mask[unchanged] = torch.rand(unchanged.sum().int().item(), device=self.device) >= self.alpha
        
        return replace_mask
    
    def generate(self, edge_index: Tensor) -> ABSizedHypergraphNegativeSamplerResult:
        positive_edge_index = edge_index[:, torch.argsort(edge_index[1])]
        negative_edge_index = torch.empty((2,0), dtype= torch.long, device= self.device)
        num_hyperedges = 0
        global_positives = torch.empty((2,0), dtype=torch.float32, device= self.device)
        global_replace_mask = torch.empty((0,), dtype=bool, device= self.device)
        global_replacement = torch.empty((0,) , dtype=torch.long, device=self.device)
    
        for _ in range(self.beta):
            local_edge_index = torch.clone(positive_edge_index)
            replace_mask = self.get_replace_mask(local_edge_index)
            probabilities = self.get_probabilities(replace_mask, positive_edge_index)
            _probabilities = torch.clone(probabilities).detach()
            if self.mode == self.Mode.BEST_EFFORT:
                pass
            elif self.mode == self.Mode.NODE_AWARE:
                nodes = local_edge_index[0, replace_mask]
                _probabilities[torch.arange(0, replace_mask.long().sum()), nodes] = 0
            elif self.mode == self.Mode.HYPEREDGE_AWARE:
                #Change the probabilities of the node in the hyperedge to 0
                for e in torch.unique(local_edge_index[1,replace_mask]):
                    edges = (local_edge_index[1, replace_mask] == 0).nonzero()
                    nodes = local_edge_index[0, local_edge_index[1] == e]
                    _probabilities[edges, nodes] = 0
            else:
                raise ValueError("Invalid mode")
            #Sampling
            _probabilities = _probabilities.sum(dim = 1, keepdim= True)
            #Avoid sampling duplicate nodes within the same hyperedge
            if self.avoide_duplicate_nodes: 
                replacement = torch.empty(replace_mask.sum().int().item(), dtype=torch.long, device= self.device)
                for e in torch.unique(local_edge_index[1, replace_mask]):
                    e_mask = local_edge_index[1, replace_mask] == e
                    replacement[e_mask] = torch.multinomial(_probabilities[e_mask], 1, replacement=False).flatten()
            else:
                replacement = torch.multinomial(_probabilities, 1, replacement=True).flatten()
            local_edge_index[0, replace_mask] = replacement
            local_edge_index[1] += num_hyperedges
            num_hyperedges = torch.max(local_edge_index[1]) + 1
            negative_edge_index = torch.cat([negative_edge_index , local_edge_index], dim = 1)
            global_positives = torch.empty((0, probabilities.shape[1]), dtype=torch.float32, device=self.device)
            global_replace_mask = torch.cat([global_replace_mask, replace_mask], dim = 0)
            global_replacement = torch.cat([global_replacement, replacement], dim = 0) 

        return ABSizedHypergraphNegativeSamplerResult(
            global_positives,
            global_replace_mask,
            global_replacement,
            self,
            torch.clone(positive_edge_index),
            negative_edge_index
        )
    
class SizedHypergraphNegativeSampler(ABSizedHypergraphNegativeSampler):
    def __init__(self, num_node, *args, **kwargs):
        super(SizedHypergraphNegativeSampler, self).__init__(num_node, 0, 1,*args, **kwargs)

class MotifHypergraphNegativeSampler(HypergraphNegativeSampler):
    """ A class Negative Sampler which use the Motif Negative Sampling 
        algorithm.
    """

    def fit(self, edge_index: Tensor, *args):
        return self
    
    def generate(self, edge_index: Tensor) -> HypergraphNegativeSamplerResult:
        num_hyperedges = edge_index[1].max().item() + 1
        sparse = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1], device=self.device),
            (self.num_node, num_hyperedges),
            device= self.device
        )
        
        # Compute sparse adjacency matrix A = sparse @ sparse.T (keeps it sparse)
        A_sparse = torch.sparse.mm(sparse, sparse.t())
        # Coalesce and get indices where A > 0 (nodes that share hyperedges)
        A_sparse = A_sparse.coalesce()
        edges = A_sparse.indices().t()  # Get non-zero edges
        
        degrees = sparse.sum(dim=0).to_dense().flatten()
        generated_hyperedges_count = 0
        generated_hyperedges = []
        unique_hyperedges = torch.unique(edge_index[1])
        
        for i in range(unique_hyperedges.shape[0]):
            while True:
                degree = degrees[torch.randint(0, degrees.shape[0], (1,))].item()
                if degree == 0 or edges.shape[0] == 0:
                    degree = max(2, int(degrees.float().mean().item()))
                
                # Start with a random edge
                f = edges[torch.randint(0, edges.shape[0], (1,))].flatten()
                
                while f.shape[0] < degree:
                    # Find nodes connected to current motif
                    # Use sparse indexing to find neighbors
                    mask = (edges[:, 0].unsqueeze(1) == f.unsqueeze(0)).any(dim=1)
                    potential_nodes = edges[mask].flatten().unique()
                    
                    # Remove already selected nodes
                    potential_nodes = potential_nodes[~torch.isin(potential_nodes, f)]
                    
                    if potential_nodes.shape[0] == 0:
                        break
                    
                    # Uniform sampling from potential nodes
                    new_node = potential_nodes[torch.randint(0, potential_nodes.shape[0], (1,))]
                    f = torch.cat([f, new_node])
                
                if f.shape[0] >= 2:  # At least 2 nodes for a valid hyperedge
                    break
            
            generated_hyperedges.append(torch.vstack([
                f,
                torch.full((1, f.shape[0]), generated_hyperedges_count, device = self.device)
            ]))
            generated_hyperedges_count += 1
        
        return HypergraphNegativeSamplerResult(
            self,
            edge_index,
            torch.cat(generated_hyperedges, dim = 1)
        )
    
class CliqueHypergraphNegativeSampler(HypergraphNegativeSampler):
    """ A class Negative Sampler which use the Clique Negative Sampling
        algorithm.
    """
    
    def fit(self, edge_index: Tensor,*args):
        return self
    
    def generate(self, edge_index: torch.Tensor) -> HypergraphNegativeSamplerResult:
        num_hyperedges = edge_index[1].max().item() + 1
        sparse = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1], device = self.device),
            (self.num_node, num_hyperedges),
            device= self.device
        )

        # Compute sparse adjacency matrix
        A_sparse = torch.sparse.mm(sparse, sparse.t()).coalesce()
        
        generated_hyperedges_count = 0
        generated_hyperedges = []
        unique_edges = torch.unique(edge_index[1])
        
        for i in range(unique_edges.shape[0]):
            while True:
                #Randomly sample an hyperedge
                hyperedge = unique_edges[torch.randint(0, unique_edges.shape[0], (1,))]
                nodes = edge_index[0, edge_index[1] == hyperedge] #Get the nodes in the hyperedge
                
                if nodes.shape[0] < 2:
                    continue
                
                #Randomly sample a node for removal
                remove_idx = torch.randint(0, nodes.shape[0], (1,))
                hyperedge_mask = torch.ones(nodes.shape[0], dtype = torch.bool, device = self.device)
                hyperedge_mask[remove_idx] = False
                
                remaining_nodes = nodes[hyperedge_mask]
                
                # Find candidate nodes: get all neighbors of remaining nodes
                candidate_nodes = []
                for node in remaining_nodes:
                    # Get nodes connected to this node via A_sparse
                    mask = A_sparse.indices()[0] == node
                    neighbors = A_sparse.indices()[1][mask]
                    candidate_nodes.append(neighbors)
                
                if len(candidate_nodes) > 0:
                    all_candidates = torch.cat(candidate_nodes).unique()
                    # Remove nodes already in hyperedge
                    all_candidates = all_candidates[~torch.isin(all_candidates, nodes)]
                    
                    if all_candidates.shape[0] > 0:
                        # Randomly select one candidate
                        new_node = all_candidates[torch.randint(0, all_candidates.shape[0], (1,))]
                        new_hyperedge = torch.cat([remaining_nodes, new_node])
                        
                        generated_hyperedges.append(torch.vstack([
                            new_hyperedge,
                            torch.full((1, new_hyperedge.shape[0]), generated_hyperedges_count, device= self.device)
                        ]))
                        generated_hyperedges_count += 1
                        break
                    else:
                        continue
                else:
                    continue
        return HypergraphNegativeSamplerResult(
            self,
            edge_index,
            torch.cat(generated_hyperedges, dim=1)
        )