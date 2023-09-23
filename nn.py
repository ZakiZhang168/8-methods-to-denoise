import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
window_size = 100
least_loss = 1e-3
num_epochs = 120
# num_epochs = 1_000

n = 1024
h = 2 * np.pi / n
t = np.arange(0, 2 * np.pi, h)

# training signal data
S = [
    1 * np.sin(3 * t) + 1 * np.cos(6 * t) - np.cos(2 * t),  # S1
    1 * np.sin(2 * t) + 1 * np.cos(7 * t) - np.cos(3 * t),
    3 * np.sin(3 * t) + 5 * np.cos(6 * t) - 2 * np.cos(1 * t)
]
RN = [
    1 * np.random.randn(n, 1),
    2 * np.random.randn(n, 1),
]

# generate another signal
S_test = 1 * np.sin(2 * t) + 1 * np.cos(7 * t) - np.cos(1 * t)
RN_test = 1 * np.random.randn(n, 1)
NS_test = RN_test.flatten() + S_test

NS = []
for s in S:
    for rn in RN:
        ns = np.copy(s)
        ns += rn.flatten()
        NS.append(ns)

# create train data and target
half_window = window_size // 2
X = []
y = []

for idx, ns in enumerate(NS):
    for i in range(half_window, len(ns) - half_window):
        window = ns[i - half_window: i + half_window]
        target = S[idx // len(RN)][i]
        X.append(window)
        y.append(target)

X = np.array(X)
y = np.array(y)

# split train data and target into train and test sets
X_train = torch.FloatTensor(X)
y_train = torch.FloatTensor(y)

# Create validation data
X_val = []
y_val = []

for i in range(half_window, len(NS_test) - half_window):
    window = NS_test[i - half_window: i + half_window]
    target = S_test[i]
    X_val.append(window)
    y_val.append(target)

X_val = torch.FloatTensor(np.array(X_val))
y_val = torch.FloatTensor(np.array(y_val))


class DenoisingModel(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=64):
        super(DenoisingModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.cnv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.cnv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


model = DenoisingModel(input_size=window_size, hidden_size1=64, hidden_size2=64)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train.unsqueeze(1))
    loss = criterion(outputs, y_train.view(-1, 1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.7f}', end='\t')

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.unsqueeze(1))
            val_loss = criterion(val_outputs, y_val.view(-1, 1))

        print(f'Validation Loss: {val_loss.item():.7f}')

        if loss < least_loss:
            break


NS_test = RN_test.flatten() + S_test  # noisy signal
X = []
for i in range(half_window, len(NS_test) - half_window):
    window = NS_test[i - half_window: i + half_window]
    X.append(window)
X = np.array(X)  # every 10 points is a sample

# use model to predict the signal
model.eval()
with torch.no_grad():
    X_test = torch.FloatTensor(X)
    X_test = X_test.unsqueeze(1)
    denoised_NS = model(X_test).numpy().flatten()
    denoised_NS = np.concatenate((S_test[:half_window], denoised_NS))
    denoised_NS = np.concatenate((denoised_NS, S_test[-half_window:]))

# plot new signal and processed signal
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title('Noisy signal')
plt.plot(t, NS_test, 'b-', label='Noisy', linewidth=2)
plt.plot(t, S_test, 'r-.', label='signal', linewidth=2)
plt.legend(['NS', 'S'])

plt.subplot(2, 1, 2)
plt.title('processed signal')
plt.plot(t, denoised_NS, 'b-', label='model', linewidth=2)
plt.plot(t, S_test, 'r-.', label='signal', linewidth=2)
plt.legend(['model', 'S'])

plt.xlabel('t')
plt.tight_layout()
plt.grid(True)
plt.show()

issave = input('Save model(name =  model_ws' + str(window_size) + '.pth)? (y/n)')
if issave == 'y':
    torch.save(model.state_dict(), 'model_ws' + str(window_size) + '.pth')
elif issave == 'n':
    pass
else:
    torch.save(model.state_dict(), issave + '.pth')
