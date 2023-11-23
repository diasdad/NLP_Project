import torch
from transformers import BertTokenizer
from transformers import BertModel
device='cuda'
pretrained=BertModel.from_pretrained('bert-base-chinese',force_download=False)
pretrained.to(device)
token=BertTokenizer.from_pretrained('bert-base-chinese',force_download=False)
vocab=token.get_vocab()


class Transformer(torch.nn.Module):
    def __init__(self,vocab_size,seq_len,hidden_size):
        super().__init__()
        # 标识当前模型是否处于tuning模式
        self.tuning = False
        # 当处于tuning模式时backbone应该属于当前模型的一部分，否则该变量为空
        self.pretrained = None
        self.fc=torch.nn.Linear(768, 10)
        self.fc_out = torch.nn.Linear(hidden_size, vocab_size)


    def forward(self,x, x_mask,x_typeid):

        if self.tuning:
            out = self.pretrained(input_ids=x,attention_mask=x_mask, token_type_ids=x_typeid).last_hidden_state[:,0]
        else:
            with torch.no_grad():
                out = pretrained(input_ids=x, attention_mask=x_mask, token_type_ids=x_typeid).last_hidden_state[:,0]

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


