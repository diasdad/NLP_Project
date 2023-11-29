from transformers import BertTokenizer
from Model import GPT2LMHeadModel
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
token=BertTokenizer.from_pretrained('bert-base-chinese',force_download=False)
vocab = token.get_vocab()
id_to_token=token.ids_to_tokens
vocab_size=len(vocab)
seq_len=80
summary_len=25
hidden_size=256
#第7章/定义数据集
device = 'cuda'
data_dir='/home/fx/PycharmProjects/NLP_Project/transformer_text_matching/train.csv'
csv_dataset = load_dataset(path='csv',
                           data_files=data_dir,
                           split='train')
dataset=csv_dataset.train_test_split(test_size=0.1)
print(dataset)

def collate_fn(data):
    # text_pairs = [(i['src'], i['tgt']) for i in data]
    seq=[i['src'] for i in data]
    target=[i['tgt'] for i in data]
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
    data_target = token.batch_encode_plus(batch_text_or_text_pairs=target,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=summary_len,
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

    target_input_ids = data_target['input_ids']
    target_attention_mask = data_target['attention_mask']
    target_token_type_ids = data_target['token_type_ids']

    #把数据移动到计算设备上
    seq_input_ids = seq_input_ids.to(device)
    seq_attention_mask  = seq_attention_mask .to(device)
    seq_token_type_ids = seq_token_type_ids.to(device)

    target_input_ids=target_input_ids.to(device)
    target_attention_mask=target_attention_mask.to(device)
    target_token_type_ids=target_token_type_ids.to(device)
    combine_input_ids=torch.cat((seq_input_ids,  target_input_ids[:,1:]), dim=1)
    combine_token_type_ids = torch.cat([seq_token_type_ids, target_token_type_ids[:, 1:] + 1], dim=1)
    # print(token.decode(data_seq['input_ids'][0]))
    # print(token.decode(data_target['input_ids'][0]))
    return combine_input_ids,combine_token_type_ids,


#第7章/数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset['train'],
                                     batch_size=60,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True,
                                     )
print(len(loader))

def train(epochs,sign=0):
    if sign==1:
        loaded_model = torch.load('model/文本摘要.model')

        # 将模型移到训练设备（例如 GPU）
        loaded_model = loaded_model.to('cuda')
        model=loaded_model
    else:
        model = GPT2LMHeadModel.from_pretrained('uer/gpt2-distil-chinese-cluecorpussmall')

    model.to(device)



    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        for i, (x,x_typeid) in enumerate(loader):
            #x = [8, 50]
            #y = [8, 51]
            #在训练时,是拿y的每一个字符输入,预测下一个字符,所以不需要最后一个字
            #[8, 50, 39]
            outputs = model(x,x,x_typeid)
            loss = outputs[0]
            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 50 == 0:
                pred = outputs[1].softmax(dim=2)
                pred = pred.argmax(2)
                pred = pred[:,seq_len:,]
                real = x[:,seq_len:,]
                lr = optim.param_groups[0]['lr']
                print("train", epoch, i, lr, loss.item())
                print("real", list(id_to_token[ids.item()] for ids in real[0,1:20]))
                print("pred",list(id_to_token[ids.item()] for ids in pred[0,0:19]))
        torch.cuda.empty_cache()

        sched.step()
        torch.save(model, 'model/文本摘要.model')


train(10)
