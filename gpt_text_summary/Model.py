import torch
from unils import *
import math
from transformers import BertTokenizer
from transformers import BertModel
from transformers import GPT2Model, GPT2Tokenizer
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2 import GPT2PreTrainedModel, GPT2Model
device='cuda'

loss_fct = CrossEntropyLoss(ignore_index=0, reduction="sum")

class GPT2LMHeadModel(GPT2PreTrainedModel):
    """GPT2模型"""

    def __init__(self, config):
        """
        初始化函数
        Args:
            config: 配置参数
        """
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    def forward(self, x,targets=None,x_typeid=None):

        out_x = self.transformer(input_ids=x)[0]
        score=self.lm_head(out_x)
        outputs = (score,)
        if targets!=None:
            targets=targets*x_typeid
            pred=score[..., :-1, :].contiguous()
            real=targets[..., 1:].contiguous()
            loss = loss_fct(pred.view(-1, pred.size(-1)), real.view(-1))
            num = real.ne(0).long().sum().item()
            loss=loss/num
            outputs = (loss,) + outputs

        return outputs

