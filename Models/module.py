import torch.nn as nn
import torch.nn.functional as F


def get_feature_extractor(backbone_type: str, input_dim):
    if backbone_type == 'MLP':
        return nn.ModuleList([
            nn.Dropout(0.1),
            nn.Linear(input_dim, 1000),   # hidden = [1000, 500, 100]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1000, 500),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(500, 100),
            # nn.ReLU()
        ])
    elif backbone_type == 'LSTM':
        return nn.ModuleList([
            nn.LSTM(input_size=input_dim, hidden_size=1000, batch_first=True),
            nn.Dropout(0.3),
            nn.LSTM(input_size=1000, hidden_size=500, batch_first=True),
            nn.Dropout(0.3),
            nn.LSTM(input_size=500, hidden_size=100, batch_first=True)
        ])
    else:
        raise NotImplementedError


def get_predictor(num_classes):
    return nn.ModuleList([
        nn.Dropout(0.1),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(100, num_classes),
    ])


