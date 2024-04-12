"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn


class PositionwiseFeedForward(nn.Module):
    # d_model = 512; hidden = 4 * d_model;
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        注意区别: 原始的MLP处理的是二维数据例如（128，512）,
        而这个PositionwiseFeedForward 中的Position体现在此处输入的不是二维的数据而是三维的例如(128, 30, 512),
        30就是序列的长度。其本质依然是对 512的特征维度进行MLP但是输入数据不同。

        此处不用改直接用是因为 pytorch的Linear实现默认最后一维是 特征维度, 直接就可以用
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
