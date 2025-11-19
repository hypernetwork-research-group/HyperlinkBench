import torch
from torch import Tensor
from .hyperlink_prediction_base import HyperlinkPredictor
from .hyperlink_prediction_result import HyperlinkPredictionResult
import torch.nn as nn

#TODO: Add PyDoc
class CommonNeighbors(HyperlinkPredictor):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.H = None        
        self.num_node = None
        self.num_hyperlink = None

    def fit(self, X, y, edge_index, *args, **kwargs):
        
        self.num_node = int(edge_index[0].max().item()) + 1
        self.num_hyperlink = int(edge_index[1].max().item()) + 1

        sparse = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1], device=self.device),
            (self.num_node, edge_index.max().item() + 1),
            device=self.device
        )

        self.H = sparse.to_dense()
        return self 
    
    def score_CN(self, H, u, v):
        return torch.dot(H[u], H[v]).item()
    
    def predict(self, edge_index: Tensor):
        if self.H is None:
            if edge_index is None:
                raise ValueError("Model not fitted. Call fit() first.")
            self.fit(None, None, edge_index)
        H = self.H
        
        CN_matrix = torch.matmul(H, H.T)

        new_edges = torch.nonzero(torch.triu(CN_matrix, diagonal=1)).T 

        return HyperlinkPredictionResult(
            edge_index=new_edges,
            device=self.device
        )
    
#TODO: Add PyDoc
"""Extract from the code of the paper: https://malllabiisc.github.io/publications/papers/nhp_cikm20.pdf"""
class NeuralHP(HyperlinkPredictor, nn.Module):
    def __init__(self, d=128, h=64, Type='s', score='mean', lam=1.0, device='cpu'):
        HyperlinkPredictor.__init__(self, device)
        nn.Module.__init__(self)

        self.H = None        
        self.X = None 
        self.device = device
        self.d, self.h = d, h
        self.Type, self.score, self.lam = Type, score, lam

        # Layers
        self.loop = None      # self-loop
        self.GCN = None       # hyperlink-aware GCN
        self.INT = nn.Linear(h, 1)
        if Type == "d":
            self.BL = nn.Linear(h, h)

        self.to(self.device)

    def fit(self, X, y, edge_index, *args, **kwargs):
        self.train()
        if self.loop is None:
            self.d = X.shape[1]
            self.loop = nn.Linear(self.d, self.h)
            self.GCN = nn.Linear(self.d, self.h)

        num_nodes = int(edge_index[0].max().item()) + 1
        num_hyperlinks = int(edge_index[1].max().item()) + 1

        H = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1], device=self.device),
            (num_nodes, num_hyperlinks),
            device=self.device
        ).to_dense()

        iX = X
        jX = X.clone()
        iAX = torch.matmul(H.T, X)
        jAX = iAX.clone()
        I = H.clone()

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        epochs = kwargs.get("epochs", 50)

        for _ in range(epochs):
            optimizer.zero_grad()
            data = {
                'iX': (iX,),
                'iAX': (iAX,),
                'jX': (jX,),
                'jAX': (jAX,),
                'I': (I,)
            }
            scores = self.forward(data, test=True)
            metrics = self.metrics(scores)
            loss = metrics['loss']
            loss.backward()
            optimizer.step()

        self.H = H.detach()
        self.X = X.detach()
        return self

    def predict(self, edge_index: Tensor):
        if self.H is None or self.X is None:
            if edge_index is None:
                raise ValueError("Model not fitted. Call fit() first.")
            self.fit(self.X, None, edge_index)
            
        H = self.H.to(self.device)
        X = self.X  

        iX = X
        jX = X.clone()
        iAX = torch.matmul(H.T, X)
        jAX = iAX.clone()
        I = H.clone()

        data = {
                'iX': (iX,),
                'iAX': (iAX,),
                'jX': (jX,),
                'jAX': (jAX,),
                'I': (I,)
            }

        self.eval()
        with torch.no_grad():
            scores = self.forward(data, test=True)
            S = scores['S'].cpu()
            S_ = scores['S_'].cpu()

        new_edges = torch.nonzero(torch.triu(torch.matmul(H, H.T), diagonal=1)).T

        return HyperlinkPredictionResult(
            edge_index=new_edges,
            device=self.device
        )

    def forward(self, data, test=False):
        iX = self._get(data, 'iX', test=test)    
        jX = self._get(data, 'jX', test=test)
        I  = self._get(data, 'I', test=test)     

        iAX = self._get(data, 'iAX', test=test)   
        jAX = self._get(data, 'jAX', test=test)
        
        iX_proj = self.loop(iX)
        jX_proj = self.loop(jX)
        
        iAX_proj = self.GCN(iAX)
        jAX_proj = self.GCN(jAX)
        
        
        iAX_agg = torch.matmul(I, iAX_proj)
        jAX_agg = torch.matmul(I, jAX_proj)
        
      
        iH = iX_proj + iAX_agg
        jH = jX_proj + jAX_agg
        
        # Scoring
        if self.score == 'mean':
            IH, JH = self._mean(I, iH), self._mean(I, jH)
        elif self.score == 'maxmin':
            IH, JH = self._maxmin(I, iH), self._maxmin(I, jH)

        S = torch.sigmoid(self.INT(IH))
        S_ = torch.sigmoid(self.INT(JH))
        
        D = {"S": S, "S_": S_}
        
        if self.Type == "d":
            IHp = self._mean((I == 1).float(), iH)
            IHn = self._mean((I == -1).float(), iH)
            Sb = torch.mm(self.BL(IHp), IHn.t()).diagonal().unsqueeze(1)
            
            JHp = self._mean((I == 1).float(), jH)
            JHn = self._mean((I == -1).float(), jH)
            Sb_ = torch.mm(self.BL(JHp), JHn.t()).diagonal().unsqueeze(1)
            
            D['Sb'], D['Sb_'] = Sb, Sb_
        
        return D


    def _mean(self, K, H):
        L = K * K
        L = L / torch.sum(L, dim=0, keepdim=True)
        return torch.mm(L.t(), H)

    def _maxmin(self, K, H):
        L = K.t()
        B = H.repeat(L.size()[0], 1, 1)
        d = B.size()[-1]
        L = L.repeat_interleave(d).view(L.size()[0], L.size()[1], d)
        LB = (L == 1).float() * B
        M = torch.max(LB, dim=1)[0]
        if self.Type == "d":
            LB = (L == -1).float() * B
        m = torch.min((LB == 0).float() * 1e4 + LB, dim=1)[0]
        return (M - m) * ((M - m > 0).float())
     
    def metrics(self, scores):
        S, S_ = scores['S'], scores['S_']
        M = torch.sum(S_) / len(S_)
        loss = torch.sum(torch.log(1 + torch.exp(M - S)))
        if 'Sb' in scores:
            Sb, Sb_ = scores['Sb'], scores['Sb_']
            Mb = torch.sum(Sb_) / len(Sb_)
            loss += self.lam * torch.sum(torch.log(1 + torch.exp(Mb - Sb)))
        S_np, S__np = S.detach().cpu().numpy(), S_.detach().cpu().numpy()
        Y = [1] * len(S_np) + [0] * len(S__np)
        Z = list(S_np) + list(S__np)
        
        return {'loss': loss}

    def _get(self, data, k, test=False):
        return data[k][0].to(self.device if not test else 'cpu')

#TODO: Add PyDoc
"""Initial code: https://github.com/srendle/libfm/blob/master/src/fm_core/fm_model.h"""
class FactorizationMachine(HyperlinkPredictor):
    def __init__(self, num_features=None, num_factors=10, reg_lambda=0.0, device='cpu'):
        super().__init__(device)
        self.num_features = num_features
        self.num_factors = num_factors
        self.reg_lambda = reg_lambda

        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = None
        self.V = None
        self.fitted = False


    def fit(self, X, y, edge_index, *args, **kwargs):
       
        if X is None:
            self.num_features = int(edge_index.max().item() + 1)
        else:
            _, self.num_features = X.shape

        self.w = nn.Parameter(torch.zeros(self.num_features, device=self.device))
        self.V = nn.Parameter(
            torch.randn(self.num_features, self.num_factors, device=self.device) * 0.01
        )

        self.fitted = True
        return self


    def predict(self, edge_index: Tensor):
        if not self.fitted:
            num_nodes = int(edge_index[0].max()) + 1
            dummy_X = torch.zeros((1, num_nodes), device=self.device)
            dummy_y = torch.zeros((1,), device=self.device)
            self.fit(dummy_X, dummy_y, edge_index)

        X = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1], device=self.device),
            (self.num_features, edge_index.shape[1]),
            device=self.device
        ).to_dense().T   

        preds = self._predict_tensor(X)

        
        positive_mask = preds.squeeze() > 0
        pos_edge_index = edge_index[:, positive_mask]

        return HyperlinkPredictionResult(
            edge_index=pos_edge_index,
            device=self.device
        )

    def _predict_tensor(self, X: Tensor) -> Tensor:
        linear_part = torch.matmul(X, self.w) + self.w0

        XV = torch.matmul(X, self.V)
        XV_square = XV.pow(2).sum(dim=1)
        X_square_V_square = torch.matmul(X.pow(2), self.V.pow(2)).sum(dim=1)
        interactions = 0.5 * (XV_square - X_square_V_square)

        return linear_part + interactions
