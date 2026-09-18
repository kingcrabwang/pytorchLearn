import torch

x_data=torch.tensor([[1.0],[2.0],[3.0]])
y_data=torch.tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(1, 1)

    def forward(self,x):
        y_pre=self.linear(x)
        return y_pre

model=LinearModel()

criterion=torch.nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):
    y_pre=model(x_data)
    loss=criterion(y_pre,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

x_text=torch.tensor([4.0])
y_test=model(x_text)
print('y_pred = ', y_test.data)