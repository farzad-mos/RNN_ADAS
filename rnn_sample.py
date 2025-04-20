# RNN Sample: Driver Drowsiness (Simulated)
import torch
from torch import nn

class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=3, hidden_size=8, batch_first=True)
        self.fc = nn.Linear(8, 2)  # Drowsy or not

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0))

model = RNNModel()
sample = torch.randn(5, 10, 3)  # batch of 5, sequence length 10, 3 features
output = model(sample)
print(output)