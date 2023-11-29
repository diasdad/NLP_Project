import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import RandomSampler
from transformers import BertTokenizer
from model import UnilmForSeq2Seq
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from unilm import UnilmConfig
import data_set
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
tb_writer = SummaryWriter()
data_dir='./data/train.json'
config_dir='./model/config.json'
output_dir='./model'
train_batch_size=25
max_pred=20
mask_prob=0.2
max_seq_length=256
skipgram_prb=0.0
skipgram_size=1
num_workers=0
num_train_epochs=10.0
gradient_accumulation_steps=1.0
learning_rate=5e-5
adam_epsilon=1e-8
warmup_proportion=0.1
seed=43
logging_steps=5
max_grad_norm=1.0
mask_whole_word=True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config=UnilmConfig.from_pretrained(config_dir)
model = UnilmForSeq2Seq(config=config)
model.to(device)
#如果从头开始训练，mask_source_words改为True， dataset第65行改为effective_length = len(tokens)
bi_uni_pipeline = [data_set.Preprocess4Seq2seq(max_pred, mask_prob, list(tokenizer.vocab.keys()),
                                               tokenizer.convert_tokens_to_ids, max_seq_length,
                                               mask_source_words=False, skipgram_prb=skipgram_prb,
                                               skipgram_size=skipgram_size,
                                               mask_whole_word=mask_whole_word, tokenizer=tokenizer)]
train_dataset = data_set.Seq2SeqDataset(
    data_dir, train_batch_size, tokenizer, max_seq_length, bi_uni_pipeline=bi_uni_pipeline)

train_sampler = RandomSampler(train_dataset, replacement=False)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers,
                                               collate_fn=data_set.batch_list_to_batch_tensors,
                                               pin_memory=False)
t_total = int(len(train_dataloader) * num_train_epochs / gradient_accumulation_steps)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup_proportion * t_total),
                                            num_training_steps=t_total)

logger.info("***** CUDA.empty_cache() *****")
torch.cuda.empty_cache()
logger.info("***** Running training *****")
logger.info("  Batch size = %d", train_batch_size)
logger.info("  Num steps = %d", t_total)

# model.train()
global_step = 0
tr_loss, logging_loss = 0.0, 0.0

model_path = 'model/Unilm预训练模型.model'
#使用预训练模型，若没有文件则从头开始训练
if os.path.exists(model_path):
    loaded_model = torch.load(model_path)
    loaded_model = loaded_model.to('cuda')
    model = loaded_model
model.train()
# 遍历每个Epoch
for i_epoch in trange(0, int(num_train_epochs), desc="Epoch", disable=False):
    # 遍历每个Batch数据

    iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)', disable=False)
    for step, batch in enumerate(iter_bar):
        batch = [t.to(device) if t is not None else None for t in batch]
        input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, _ = batch
        # 损失计算
        masked_lm_loss = model(input_ids, segment_ids, input_mask, lm_label_ids,
                               masked_pos=masked_pos, masked_weights=masked_weights)

        loss = masked_lm_loss
        tr_loss += loss.item()
        iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        # 损失回传
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        # 模型参数优化
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            if logging_steps > 0 and global_step % logging_steps == 0:
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) /logging_steps, global_step)
                logging_loss = tr_loss
    torch.save(model, 'model/Unilm预训练模型.model')
    # 每一个Epoch进行模型保存
    # logger.info("** ** * Saving fine-tuned model and optimizer ** ** * ")
    # output_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
    # model_to_save = model.module if hasattr(model, "module") else model
    # model_to_save.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)
    # config.save_pretrained(output_dir)
    torch.cuda.empty_cache()