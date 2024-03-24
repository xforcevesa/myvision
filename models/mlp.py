from torch import nn


class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ELU()
        )
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.ELU()
            ) for _ in range(num_layers - 1)
        ])
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.fc1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc2(x)
        return x
