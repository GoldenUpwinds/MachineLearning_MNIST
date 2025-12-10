import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 构建数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
)

# 加载数据集
train_dataset = datasets.MNIST(
    root="../data", train=True, download=True, transform=transform
)

test_datset = datasets.MNIST(
    root="../data", train=False, download=True, transform=transform
)

# Dataloader
batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_datset, batch_size=batch_size, shuffle=False)


# 线性模型
class LinearMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)


model = LinearMNIST()

# === MSE损失 ===
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练
epoches = 10
for epoch in range(epoches):
    model.train()
    loss_all = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # === 关键修改：标签转为 one-hot ===
        y_onehot = torch.nn.functional.one_hot(y_batch, num_classes=10).float()

        # 前向
        logits = model(X_batch)

        # MSELoss 输入必须同形状 (batch, 10)
        loss = criterion(logits, y_onehot)

        # 反向与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all += loss.item() * X_batch.size(0)

    avg_loss = loss_all / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epoches}, Train Loss = {avg_loss:.4f}")


# === 准确率评估 ===
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total


train_acc = evaluate(model, train_loader)
test_acc = evaluate(model, test_loader)

print(f"PyTorch MSE Linear Model Train Acc: {train_acc:.4f}")
print(f"PyTorch MSE Linear Model Test Acc: {test_acc:.4f}")
