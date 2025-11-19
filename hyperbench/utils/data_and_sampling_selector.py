from ..hyperlink_prediction.datasets import ARBDataset, IMDBHypergraphDataset, ARXIVHypergraphDataset, COURSERAHypergraphDataset, CHLPBaseDataset
from ..negative_sampling.hypergraph_negative_sampling_algorithm import SizedHypergraphNegativeSampler, MotifHypergraphNegativeSampler, CliqueHypergraphNegativeSampler, HypergraphNegativeSampler
from ..hyperlink_prediction.models import CommonNeighbors, NeuralHP, FactorizationMachine, HyperlinkPredictor

def setNegativeSamplingAlgorithm(ns_algorithm: str, num_node: int, alpha: float|int, beta: int):
    ns_method : HypergraphNegativeSampler
    match(ns_algorithm):
        case 'SizedHypergraphNegativeSampler':
            ns_method = SizedHypergraphNegativeSampler(num_node, alpha= alpha, beta= beta)
        case 'MotifHypergraphNegativeSampler': 
            ns_method = MotifHypergraphNegativeSampler(num_node)
        case 'CliqueHypergraphNegativeSampler':
            ns_method = CliqueHypergraphNegativeSampler(num_node)
    
    return ns_method

def setHyperlinkPredictionAlgorithm(hlp_algorithm: str):
    hlp_method: HyperlinkPredictor
    match(hlp_algorithm):
        case 'CommonNeighbors':
            hlp_method = CommonNeighbors()
        case 'NeuralHP':
            hlp_method = NeuralHP()
        case 'FactorizationMachine':
            hlp_method = FactorizationMachine()
    return hlp_method

def select_dataset(ds: str, pre_transform):
    
    dataset : ARBDataset
    if ds in ARBDataset.GDRIVE_IDs.keys():
        dataset = ARBDataset(ds, pre_transform= pre_transform)
    else:
        dataset : CHLPBaseDataset
        match(ds):
            case 'IMDB': 
                dataset = IMDBHypergraphDataset("./data", pre_transform= pre_transform)
            case 'ARXIV':
                dataset = ARXIVHypergraphDataset("./data", pre_transform = pre_transform)
            case 'COURSERA':
                dataset = COURSERAHypergraphDataset("./data", pre_transform = pre_transform)
    
    return dataset