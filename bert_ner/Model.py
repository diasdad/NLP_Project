import torch
import math
from transformers import  AutoTokenizer
from transformers import  AutoModel
from transformers import BertTokenizer
from transformers import BertModel
device='cuda'
pretrained=AutoModel.from_pretrained('hfl/rbt3',force_download=False)

pretrained.to(device)
token=AutoTokenizer.from_pretrained('hfl/rbt3',force_download=False)
vocab=token.get_vocab()


class Transformer(torch.nn.Module):
    def __init__(self,vocab_size,seq_len,hidden_size):
        super().__init__()
        # 标识当前模型是否处于tuning模式
        self.tuning = False
        # 当处于tuning模式时backbone应该属于当前模型的一部分，否则该变量为空
        self.pretrained = None
        self.fc=torch.nn.Linear(hidden_size,8)
        self.lstm=torch.nn.LSTM(768,hidden_size,batch_first=True)


    def forward(self,x, x_mask,x_typeid):
        #[b, 1, 50, 50]
        batch_size, seq_len = x.size()
        if self.tuning:
            out = self.pretrained(input_ids=x,attention_mask=x_mask, token_type_ids=x_typeid).last_hidden_state
        else:
            with torch.no_grad():
                out = pretrained(input_ids=x, attention_mask=x_mask, token_type_ids=x_typeid).last_hidden_state

        out,_=self.lstm(out)
        score=self.fc(out)
        return score

    def fine_tuning(self, tuning):
        self.tuning = tuning
        # tuning模式时，训练backbone的参数
        if tuning:
            for i in pretrained.parameters():
                i.requires_grad = True

            pretrained.train()
            self.pretrained = pretrained
        # 非tuning模式时，不训练backbone的参数
        else:
            for i in pretrained.parameters():
                i.requires_grad_(False)

            pretrained.eval()
            self.pretrained = None

