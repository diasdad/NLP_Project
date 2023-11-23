import jieba
import random
import pandas as pd
from torchtext import data

def read_data(data_dir):

    # LABEL=data.LabelField()
    SEQ1=data.Field(tokenize=jieba.lcut,include_lengths=True,lower=True,init_token='<sos>', eos_token='<eos>')
    SEQ2=data.Field(tokenize=jieba.lcut,include_lengths=True,lower=True,init_token='<sos>', eos_token='<eos>')
    fields=[('seq',SEQ1),('target',SEQ2)]
    movieDateset=data.TabularDataset(path=data_dir
                                         ,format='CSV'
                                         ,fields=fields
                                         ,skip_header=False
                                         )
    train, test, val = movieDateset.split(split_ratio=[0.8, 0.1, 0.1])
    a = len(train)
    print(vars(train.examples[10]))
    vocab_size = 10000
    SEQ1.build_vocab(train, max_size=vocab_size)
    SEQ2.vocab = SEQ1.vocab  # 使用SEQ1的词汇表来初始化SEQ2的词汇表
    print(SEQ1.vocab.freqs.most_common(10))
    print(SEQ2.vocab.freqs.most_common(10))
    print(SEQ1.vocab.stoi['<sos>'])
    print(SEQ1.vocab.stoi['<eos>'])
    print(SEQ1.vocab.stoi['<unk>'])
    print(SEQ1.vocab.stoi['<pad>'])
    return train,test,val,SEQ1,SEQ2