import torch
a = torch.Tensor(2, 2)
a
b = torch.tensor(2, 2)
b = torch.tensor(2)
b
b = a.double()
b
c = a.type(torch.DoubleTensor)
c
d = a.type_as(b)
d
cls
a = torch.Tensor(2, 2)
a
b = torch.DoubleTensor(2, 2)
b
c = torch.Tensor([1, 4, 5, [4, 9]])
c = torch.Tensor([1, 4, 5])
c
d = torch.zeros(2, 2, 2)
d
e = torch.ones(2, 2)
e
f = torch.eye(2, 2)
f
f = torch.eye(2, 4)
f
g = torch.rand(2, 2)
g
cls
torch.randn(2, 2)
torch.arange(1, 100, 10)
torch.linspace(1, 6, 2)
torch.randperm(4)
torch.randperm(4)
torch.randperm(4)
torch.randperm(4)
torch.randperm(4)
k = torch.tensor([283, 84])
k
k.shape
k.size()
k.numel()
a.nelement()
a
cls
a = torch.Tensor([[1, 2],[3, 4]])
b = torch.Tensor([[1., 2.],[3., 4.]])
b
torch.cat(a, b)
torch.cat((a, b), 0)
torch.cat((a, b), 1)
torch.cat((a, b), 1).size()
torch.cat((a, b), 0).size()
torch.stack((a, b), 0)
torch.stack((a, b), 0).size()
torch.stack((a, b), 1)
torch.stack((a, b), 2)
torch.stack((a, b), 2).size()
cls
a = torch.Tensor([[1 ,2, 3], [4, 5, 6]])
torch.chunk(a, 2, 0)
torch.chunk(a, 2, 1)
torch.split(a, 2, 0)
torch.split(a, 2, 1)
%hist
%hist -f torch_test.py
