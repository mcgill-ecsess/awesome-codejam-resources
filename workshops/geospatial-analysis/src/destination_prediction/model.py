import torch
from torch import nn
import torch.nn.functional as F
#from utils import device
from src.destination_prediction.utils import device

class DestinationLSTM(nn.Module):
    
    def __init__(self, hidden_size=128, num_layers=1):
        super(DestinationLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size=2, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers)
        
        self.long_regressor = nn.Linear(in_features=hidden_size, 
                                        out_features=1)
        self.lat_regressor = nn.Linear(in_features=hidden_size, 
                                       out_features=1)
        
        self.hidden0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size) * 0.05)
        self.cell0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size) * 0.05)
        
        # initializing parameters with Glorot
        #nn.init.xavier_uniform_(self.hidden0)
        #nn.init.xavier_uniform_(self.cell0)
        nn.init.xavier_uniform_(self.long_regressor.weight)
        nn.init.xavier_uniform_(self.lat_regressor.weight)

    def forward(self, x):
        
        self._init_hidden(batch_size=x.shape[1])
        
        x, (self.hidden, self.cell) = self.lstm(x, (self.hidden, self.cell))
        
        
        
        long = self.long_regressor(x[-1])
        lat = self.lat_regressor(x[-1])

        return (lat, long)

    def _init_hidden(self, batch_size):
        
        self.hidden = self.hidden0.clone().repeat(1, batch_size, 1).to(device)
        self.cell = self.cell0.clone().repeat(1, batch_size, 1).to(device)

class DestinationLSTMClf(nn.Module):
    
    def __init__(self, graph, hidden_size=128, num_layers=1):
        super(DestinationLSTMClf, self).__init__()
        
        self.graph = graph
        
        self.lstm = nn.LSTM(input_size=2, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers)
        
        self.clf = nn.Linear(in_features=hidden_size,
                             out_features=len(graph.nodes))
        
        self.hidden0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size) * 0.05)
        self.cell0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size) * 0.05)
        
        # initializing parameters with Glorot
        #nn.init.xavier_uniform_(self.hidden0)
        #nn.init.xavier_uniform_(self.cell0)
        nn.init.xavier_uniform_(self.clf.weight)

    def forward(self, x):
        
        self._init_hidden(batch_size=x.shape[1])
        
        x, (self.hidden, self.cell) = self.lstm(x, (self.hidden, self.cell))
        out = self.clf(x[-1])

        return out

    def _init_hidden(self, batch_size):
        
        self.hidden = self.hidden0.clone().repeat(1, batch_size, 1).to(device)
        self.cell = self.cell0.clone().repeat(1, batch_size, 1).to(device)

        
        
if __name__ == "__main__":
    
    from utils import device
    
    seq_len = 10
    batch_size = 128
    input_size = 2
        
    inputs = torch.randn(seq_len, batch_size, input_size).to(device)
    model = DestinationLSTM().to(device)
    out = model(inputs)
