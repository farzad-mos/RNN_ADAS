# RNN Tutorial for ADAS

## What is an RNN?
Recurrent Neural Networks are suited for sequence modeling tasks like time series, sensor fusion, or voice command recognition.

## ADAS Use Cases:
- Drowsiness Detection from time-series data
- Driver behavior prediction
- Voice command understanding

## Sample Python Code:
```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
```