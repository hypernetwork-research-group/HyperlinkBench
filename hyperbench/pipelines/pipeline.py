import argparse

def execute():
    parser = argparse.ArgumentParser(description="Insert dataset_name, insert negative_sampling method")
    parser.add_argument('--dataset_name', type=str, help="The dataset's name, possible dataset's name: \nIMDB,\nCOURSERA,\nARXIV", required=True)
    parser.add_argument('--negative_sampling', type=str, help="negative sampling method to use, possible methods: \n SizedHypergraphNegativeSampler,\nMotifHypergraphNegativeSampler,\nCliqueHypergraphNegativeSampler", required=True)
    parser.add_argument('--alpha', type= float, help="first parameter for the method SizedNegativeSampler", default=0.5)
    parser.add_argument('--beta', type= int, help="second parameter for the method SizedNegativeSampler", default=1)
    parser.add_argument('--hlp_method', type=str, help="hyperlink prediction method to use, possible methods: \nCommonNeighbors,\nNeuralHP, \nFactorizationMachine", required=True)
    parser.add_argument('--output_path', type=str, help="Path to save the results", default="./results")
    parser.add_argument('--random_seed', type=int, help="Random seed for reproducibility", default=None)
    parser.add_argument('--test', type=bool, help="If true, runs in test mode", default=False)
    args = parser.parse_args()
    dataset_name= args.dataset_name
    negative_method = args.negative_sampling
    alpha = args.alpha
    beta = args.beta
    hlp_method = args.hlp_method
    output_path = args.output_path
    random_seed = args.random_seed
    test = args.test

    import torch
    import numpy as np
    import seaborn as sns
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import time
    from random import randint, seed
    from ..hyperlink_prediction.loader.dataloader import DatasetLoader
    from ..utils.data_and_sampling_selector import setNegativeSamplingAlgorithm, select_dataset, setHyperlinkPredictionAlgorithm
    from ..utils.hyperlink_train_test_split import train_test_split
    from torch_geometric.nn import HypergraphConv
    from tqdm.auto import trange, tqdm
    from torch_geometric.data.hypergraph_data import HyperGraphData
    from torch_geometric.nn.aggr import MeanAggregation
    from torch.utils.tensorboard import SummaryWriter
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        seed(random_seed)

    def sensivity_specifivity_cutoff(y_true, y_score):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        idx = np.argmax( tpr - fpr)

        return thresholds[idx]

    now = time.strftime("%Y%m%d-%H%M%S")
    execution_name = f"{now}_{dataset_name}"
    writer = SummaryWriter(f"./logs/{execution_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def pre_transform(data: HyperGraphData):
        data.edge_index = data.edge_index[:, torch.isin(data.edge_index[1], (data.edge_index[1].bincount() > 1).nonzero())]
        unique, inverse = data.edge_index[1].unique(return_inverse = True)
        data.edge_attr = data.edge_attr[unique]
        data.edge_index[1] = inverse
        return data
    
    dataset = select_dataset(dataset_name, pre_transform= pre_transform)
    if test:
        reduction = min(1000, dataset._data.num_edges)
        edge_index = dataset._data.edge_index[:, :reduction].clone()
        edge_attr = dataset._data.edge_attr[:reduction].clone()
        nodes_present = torch.unique(edge_index[0]).sort()[0]
        num_nodes = edge_index.max().item() + 1
        mapping = -torch.ones(num_nodes, dtype=torch.long)
        mapping[nodes_present] = torch.arange(len(nodes_present))
        edge_index = mapping[edge_index]
        test_data = HyperGraphData(
            x=dataset._data.x[nodes_present].clone(),
            edge_index=edge_index,
            edge_attr=edge_attr.clone(),
            num_nodes=len(nodes_present),
        )
        dataset._data = test_data

    test_size = 0.2
    val_size = 0.0
    train_size = 1 - (test_size + val_size)
    train_dataset, test_dataset, _, _, _, _ = train_test_split(dataset, test_size = test_size)

    loader = DatasetLoader(
        train_dataset, 
        negative_method,
        alpha,
        beta,
        dataset._data.num_nodes,
        batch_size=4000, 
        shuffle=True, 
        drop_last = True
    )

    class Model(nn.Module):
        
        def __init__(self, 
                    in_channels: int,
                    hidden_channels: int,
                    out_channels: int,
                    num_layers: int = 1):
            super(Model, self).__init__()
            self.dropout = nn.Dropout(0.3)
            self.activation = nn.LeakyReLU()
            self.in_norm = nn.LayerNorm(in_channels)
            self.in_proj = nn.Linear(in_channels, hidden_channels)
            self.e_proj = None
            self.e_norm = None

            for i in range(num_layers):
                setattr(self, f"n_norm_{i}", nn.LayerNorm(hidden_channels))
                setattr(self, f"e_norm_{i}", nn.LayerNorm(hidden_channels))
                setattr(self, f"hgconv_{i}",HypergraphConv(
                    hidden_channels,
                    hidden_channels,
                    use_attention=True,
                    concat=False,
                    heads=1
                ))
                setattr(self, f"skip_{i}", nn.Linear(hidden_channels, hidden_channels))
            self.num_layers = num_layers

            self.aggr = MeanAggregation()
            self.linear = nn.Linear(hidden_channels, hidden_channels)
        
        def forward(self, x, x_e, edge_index):
            x = self.in_norm(x)
            x = self.in_proj(x)
            x = self.activation(x)
            x = self.dropout(x)
            if self.e_norm is None:
                self.e_norm = nn.LayerNorm(x_e.size(-1)).to(x_e.device)

            x_e = x_e.to(dtype=self.e_norm.weight.dtype, device=self.e_norm.weight.device)
            x_e = self.e_norm(x_e)
            if self.e_proj is None:
                self.e_proj = nn.Linear(x_e.size(-1), self.in_proj.out_features).to(x_e.device)
            x_e = self.e_proj(x_e)
            x_e = self.activation(x_e)
            x_e = self.dropout(x_e)

            for i in range(self.num_layers):
                n_norm = getattr(self, f"n_norm_{i}")
                e_norm = getattr(self, f"e_norm_{i}")
                hgconv = getattr(self, f"hgconv_{i}")
                skip = getattr(self, f"skip_{i}")
                x = n_norm(x)
                x_e = e_norm(x_e)
                x = self.activation(hgconv(x, edge_index, hyperedge_attr = x_e)) + skip(x)

            x = self.aggr(x[edge_index[0]], edge_index[1])
            x = self.linear(x)

            return x
        
    model = Model(
        in_channels = dataset.num_features,
        hidden_channels = 256,
        out_channels= 1
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()
    test_criterion = torch.nn.BCELoss()

    negative_hypergraph = setNegativeSamplingAlgorithm(negative_method, test_dataset._data.num_nodes, alpha, beta).generate(test_dataset._data.edge_index)
    edge_index_test = test_dataset._data.edge_index.clone()
    test_dataset.y = torch.vstack((
        torch.ones((test_dataset._data.edge_index[1].max() + 1, 1)),
        torch.zeros((edge_index_test[1].max() + 1, 1))
    ))

    test_dataset_ = HyperGraphData(
        x = test_dataset.x,
        edge_index= negative_hypergraph.edge_index,
        edge_attr= torch.vstack((test_dataset.edge_attr, test_dataset.edge_attr)),
        y = test_dataset.y,
        num_nodes = test_dataset._data.num_nodes
    )
    print(test_dataset_.edge_attrs)

    hlp_method = setHyperlinkPredictionAlgorithm(hlp_method)
    for epoch in trange(150):
        model.train()
        optimizer.zero_grad()
        for i, h in tqdm(enumerate(loader), leave = False):
            negative_sampler = setNegativeSamplingAlgorithm(negative_method, h.num_nodes, alpha, beta)
            negative_test = negative_sampler.generate(h.edge_index)
            hlp_method.X = h.x
            hlp_result = hlp_method.predict(negative_test.edge_index)

            y_pos = torch.ones(hlp_result.edge_index.size(1), 1)
            y_neg = torch.zeros(negative_test.edge_index.size(1), 1)

            combined_edge_index = torch.hstack([hlp_result.edge_index, negative_test.edge_index])
            combined_y = torch.vstack([y_pos, y_neg])

            pos_edge_attr = torch.ones((hlp_result.edge_index.size(1), h.edge_attr.size(1)))
            neg_edge_attr = torch.zeros((negative_test.edge_index.size(1), h.edge_attr.size(1)))
            combined_edge_attr = torch.vstack([pos_edge_attr, neg_edge_attr])

            h_ = HyperGraphData(
                x=h.x,
                edge_index=combined_edge_index,
                edge_attr=combined_edge_attr,
                y=combined_y,
                num_nodes=h.num_nodes
            )

            y_train = model(h_.x, h_.edge_attr, h_.edge_index)
            if y_train.size(1) != 1:
                y_train = y_train[:, 0:1]
            loss = criterion(y_train, h_.y)
            loss.backward()
            writer.add_scalar("Loss/train", loss.item(), epoch * len(loader) + i)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            y_test = model(test_dataset_.x, test_dataset_.edge_attr, test_dataset_.edge_index)
            y_test = torch.sigmoid(y_test)
            if y_test.size(1) != 1:
                y_test = y_test[:, 0:1]
            loss = test_criterion(y_test, test_dataset_.y)
            writer.add_scalar("Loss/test", loss.item(), epoch)
            roc_auc = roc_auc_score(test_dataset_.y.cpu().numpy(), y_test.cpu().numpy())
            writer.add_scalar("ROC_AUC/test", roc_auc, epoch)
        
    cutoff = sensivity_specifivity_cutoff(test_dataset_.y.cpu().numpy(), y_test.cpu().numpy())
    cm = confusion_matrix(
        test_dataset_.y.cpu().numpy(),
        (y_test > cutoff).cpu().numpy(),
        labels=[0, 1],
        normalize='true'
    )

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Normalized Confusion Matrix")
    plt.savefig(f"{output_path}/{execution_name}_confusion_matrix.png")

    print(f"Log folder name: {output_path}/{execution_name}")

    from pathlib import Path
    import csv
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    from sklearn.metrics import average_precision_score
    result_file = Path(f"{output_path}/exp.csv")
    file_exists = result_file.is_file()
    with open(result_file, mode='a', newline='') as file:
        my_writer = csv.writer(file)
        if not file_exists:
            my_writer.writerow(["name", "hlp_method", "negative_sampling", "random_seed", "train_ratio", "val_ratio", "test_ratio", "auc", "aupr", "f1", "accuracy", "precision", "recall"])
        auc = roc_auc_score(test_dataset_.y.cpu().numpy(), y_test.cpu().numpy())
        aupr = average_precision_score(test_dataset_.y.cpu().numpy(), y_test.cpu().numpy())
        f1 = f1_score(test_dataset_.y.cpu().numpy(), (y_test > cutoff).cpu().numpy())
        accuracy = accuracy_score(test_dataset_.y.cpu().numpy(), (y_test > cutoff).cpu().numpy())
        precision = precision_score(test_dataset_.y.cpu().numpy(), (y_test > cutoff).cpu().numpy())
        recall = recall_score(test_dataset_.y.cpu().numpy(), (y_test > cutoff).cpu().numpy())
        my_writer.writerow([execution_name, hlp_method, negative_method, random_seed, train_size, val_size, test_size, auc, aupr, f1, accuracy, precision, recall])

    writer.close()