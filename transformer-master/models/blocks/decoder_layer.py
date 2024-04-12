"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, trg_x, encoder_out, trg_mask, src_mask):
        """
        注意区分:
        enc 是前面 encoder层的输出结果
        dec 是目标数据, 或者说是上一个 decoder的输出

        输入为 src_mask的其实只是去除 pad 占位符
        输入为 trg_mask才是原论文的 mask
        详情看 transformer.py
        """
        # 1. compute self attention (masked Self-Attention)
        _x = trg_x
        """
        注意此处是训练阶段的，在预测阶段，k和v不是输入数据
        而是之前预测值的concat (pre_pred_trg_x)
        self.self_attention(q=trg_x, k=pre_pred_trg_x, v=pre_pred_trg_x, mask=trg_mask)
        """
        x = self.self_attention(q=trg_x, k=trg_x, v=trg_x, mask=trg_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if encoder_out is not None:
            # 3. compute encoder - decoder attention
            # x 来自 dec处理的结果
            _x = x
            x = self.enc_dec_attention(q=x, k=encoder_out, v=encoder_out, mask=src_mask)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
