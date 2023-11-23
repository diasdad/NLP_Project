
from Model import Transformer
from transformers import  AutoTokenizer
from transformers import  AutoModel
from datasets import load_dataset,load_from_disk
from torch.utils.data import DataLoader
from unils import *
token=AutoTokenizer.from_pretrained('hfl/rbt3',force_download=False)
vocab = token.get_vocab()
vocab_size=len(vocab)
seq_len=50
hidden_size=256
#第7章/定义数据集
device = 'cuda'
pretrained=AutoModel.from_pretrained('hfl/rbt3',force_download=False)
pretrained.to(device)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset =load_from_disk(dataset_path='peoples_daily_ner')[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        tokens=self.dataset[i]['tokens']
        labels=self.dataset[i]['ner_tags']
        return tokens,labels
dataset_train=Dataset('train')
dataset_test=Dataset('test')
def collate_fn(data):

    tokens=[i[0]for i in data]
    labels=[i[1] for i in data]
    #编码
    data_seq = token.batch_encode_plus(tokens,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=seq_len,
                                   return_tensors='pt',
                                   return_length=True,
                                   return_attention_mask=True,
                                   return_token_type_ids=True,
                                   return_special_tokens_mask=True,
                                   is_split_into_words=True
                                    )


    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    seq_input_ids = data_seq['input_ids']
    seq_attention_mask = data_seq['attention_mask']
    seq_token_type_ids = data_seq['token_type_ids']


    #把数据移动到计算设备上
    seq_input_ids = seq_input_ids.to(device)
    seq_attention_mask  = seq_attention_mask .to(device)
    seq_token_type_ids = seq_token_type_ids.to(device)

    lens=seq_input_ids.shape[1]
    for i in range(len(labels)):
        labels[i]=[7]+labels[i]
        labels[i]+=[7]*lens
        labels[i]=labels[i][:lens]
    labels = torch.tensor(labels )
    labels  = labels .to(device)

    # print(token.decode(data_seq['input_ids'][0]))
    # print(token.decode(data_target['input_ids'][0]))
    return seq_input_ids,seq_attention_mask, seq_token_type_ids ,labels
def reshape_and_remove_pad(outs, labels, attention_mask):

    outs = outs.reshape(-1, 8)
    labels = labels.reshape(-1)
    select = attention_mask.reshape(-1) == 1
    outs = outs[select]
    labels = labels[select]

    return outs, labels


#第7章/数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                     batch_size=150,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True,
                                     )
test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                     batch_size=150,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True,
                                     )

model=Transformer(vocab_size,seq_len,hidden_size).to(device)
# model.tuning=True
# model.fine_tuning(True)
def train(epochs):
    loss_func = torch.nn.CrossEntropyLoss()
    if model.tuning==True:
        lr=3e-5
    else:
        lr=3e-4
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)


    for epoch in range(epochs):
        model.train()
        for i, (x, x_mask,x_typeid,y) in enumerate(loader):
            score = model(x,x_mask,x_typeid)
            score = score.softmax(dim=2)
            score,y=reshape_and_remove_pad(score, y, x_mask)

            loss = loss_func(score, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 50 == 0:
                pred = score.argmax(dim=1)
                select=y!=0
                pred=pred[select]
                y=y[select]
                correct = (pred == y).sum().item()
                accuracy = correct / len(pred)
                lr = optim.param_groups[0]['lr']
                print("train",epoch, i, lr, loss.item(), accuracy)
        sched.step()


#

train(1)
model.tuning=True
model.fine_tuning(True)
train(5)
def test():
    model.eval()
    for i, (x, x_mask, x_typeid, y) in enumerate(test_loader):

        score = model(x, x_mask, x_typeid, y)

        # [8, 50, 39] -> [400, 39]
        score = score.softmax(dim=1)
        score, y = reshape_and_remove_pad(score, y, x_mask)
        if i % 200 == 0:
            # [select, 39] -> [select]
            pred = score.argmax(dim=1)
            select = y != 0
            pred = pred[select]
            y = y[select]
            correct = (pred == y).sum().item()
            accuracy = correct / len(pred)
            print("test", i, accuracy)

test()

