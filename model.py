import torch
import torch.nn as nn

class FreightRateModel(nn.Module):
    """
    Fully connected nearual network to predict Freight Rate
    """

    def __init__(self, input_size:int, hidden_layers = [256, 128, 64], dropout=0.2):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, hidden_layers[0]),
            nn.ReLU(),
            #nn.BatchNorm1d(hidden_layers[0]),
            nn.Dropout(dropout),

            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            #nn.BatchNorm1d(hidden_layers[1]),
            nn.Dropout(dropout),

            nn.Linear(hidden_layers[1], hidden_layers[2]),
            nn.ReLU(),
            
            nn.Linear(hidden_layers[2],1)
        )

    def forward(self,x):
        return self.network(x)