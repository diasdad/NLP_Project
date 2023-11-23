from transformers import BertTokenizer
from transformers import BertModel
from Model import Transformer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
token=BertTokenizer.from_pretrained('bert-base-chinese',force_download=False)
vocab = token.get_vocab()
id_to_token = token.ids_to_tokens
vocab_size=len(vocab)
seq_len=100
hidden_size=256
#第7章/定义数据集
device = 'cuda'
pretrained=BertModel.from_pretrained('bert-base-chinese',force_download=False)
pretrained.to(device)
data_dir='/home/fx/PycharmProjects/NLP_Project/bert_multclassfication/mult.csv'

csv_dataset = load_dataset(path='csv',
                           data_files=data_dir,
                           split='train')
dataset=csv_dataset.train_test_split(test_size=0.1)
print(dataset)

def collate_fn(data):
    # text_pairs = [(i['src'], i['tgt']) for i in data]
    seq=[i['seq'] for i in data]
    target=[i['target'] for i in data]
    #编码
    data_seq = token.batch_encode_plus(batch_text_or_text_pairs=seq,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=seq_len,
                                   return_tensors='pt',
                                   return_length=True,
                                   return_attention_mask=True,
                                   return_token_type_ids=True,
                                   return_special_tokens_mask=True)


    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    seq_input_ids = data_seq['input_ids']
    seq_attention_mask = data_seq['attention_mask']
    seq_token_type_ids = data_seq['token_type_ids']


    #把数据移动到计算设备上
    seq_input_ids = seq_input_ids.to(device)
    seq_attention_mask  = seq_attention_mask .to(device)
    seq_token_type_ids = seq_token_type_ids.to(device)
    target = torch.tensor(target)
    target = target.to(device)

    # print(token.decode(data_seq['input_ids'][0]))
    # print(token.decode(data_target['input_ids'][0]))
    return seq_input_ids,seq_attention_mask, seq_token_type_ids ,target


#第7章/数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset['train'],
                                     batch_size=150,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True,
                                     )
test_loader = torch.utils.data.DataLoader(dataset=dataset['test'],
                                     batch_size=150,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True,
                                     )

model=Transformer(vocab_size,seq_len,hidden_size).to(device)
# model.tuning=True
# model.fine_tuning(True)
def train():
    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)


    for epoch in range(10):
        model.train()
        for i, (x, x_mask,x_typeid,y) in enumerate(loader):

            score = model(x,x_mask,x_typeid)
            score = score.softmax(dim=1)
            loss = loss_func(score, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 50 == 0:
                #[select, 39] -> [select]
                pred = score.argmax(dim=1)
                correct = (pred == y).sum().item()
                accuracy = correct / len(pred)
                lr = optim.param_groups[0]['lr']
                print("train",epoch, i, lr, loss.item(), accuracy)
        sched.step()



train()



def test():
    model.eval()
    for i, (x, x_mask, x_typeid, y) in enumerate(test_loader):

        score = model(x, x_mask, x_typeid, y)

        score = score.softmax(dim=1)

        if i % 200 == 0:
            # [select, 39] -> [select]
            pred = score.argmax(dim=1)
            correct = (pred == y).sum().item()
            accuracy = correct / len(pred)
            print("test", i, accuracy)

test()
