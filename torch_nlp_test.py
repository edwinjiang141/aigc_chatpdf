import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext

from torchtext.data.utils import get_tokenizer
from pyitcast.transformer import TransformerModel


#将数据进行一个语料库的封装
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token = '<sos>',
                            eos_token = '<eo>',
                            lower = True)
print(TEXT)

train_txt,vol_txt,test_txt = torchtext.datasets.WikiText2.splits(TEXT)
print(test_txt.examples[0].text[:10])

#将训练集文本构建一个vocab对象，方便使用它的stoi的方法来统计不重复的词汇总数
TEXT.build_vocab(train_txt)
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

#构建用于模型输入的批次化数据
def batchify(data,batch_size):
    #data 代表 train_txt,vol_txt,test_txt
    #第一步使用numericalize将单词映射成连续数字
    data = TEXT.numericalize([data.examples[0].text])
    #取得需要经过对少次的batch_size后能够遍历完所有的数据
    nbranch = data.size(0) // batch_size
    #利用narrow方法对数据进行切割
    #第一个参数代表横轴还是纵轴进行切割，0代表横轴  1代表纵轴
    #第二、三个参数代表起始、终止位置
    data = data.narrow(0,0,nbranch*batch_size)

    #对data的形状进行转变
    data = data.view(batch_size,-1).t().contiguous()
    return data.to(device)

#设置训练数据的批次大小
batch_size = 20
#设置验证数据和测试数据大小
eval_batch_size = 10

#获得各项数据
train_data = batchify(train_txt,batch_size)
val_data = batchify(vol_txt,eval_batch_size)
test_data = batchify(test_txt,eval_batch_size)

#设定句子的最大长度
bptt = 35

def get_batch(source,i):
    #source代表的各种数据，i代表批次数
    seq_len = min(bptt,len(source)-i-1)

    data = source[i:i+seq_len]

    target = source[i+1:i+1+seq_len].view(-1)
    return data,target

# source = test_data
# i = 1
# x,y = get_batch(source,i)
# print('data:',x)
# print('target:',y)

#构建训练和评估函数

ntokens = len(TEXT.vocab.stoi)
emsize = 200
#前馈全连接的节点数
nhid = 200
#设置编码器层的层数
nlayers =2
#设置多头注意力中的头数
nhead = 2
dropout = 0.2

model = TransformerModel(ntokens,emsize,nhead,nhid,nlayers,dropout).to(device)

#设定损失函数
criterion = nn.CrossEntropyLoss()
#设置学习率
lr = 5.0
#设置优化器
optimizer = torch.optim.SGD(model.parameters(),lr=lr)
#定义学习率的调整器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1.0,gamma=0.95)

#训练代码分析
import time
#构建训练函数
def train():
    #开始训练模式
    model.train()
    total_loss = 0.
    start_time = time.time()
    log_interval = 200
    #遍历训练数据
    for batch,i in enumerate(range(0,train_data.size(0)-1,bptt)):
        data,target = get_batch(train_data,i)
        optimizer.zero_grad()
        #通过model预测输出
        output = model(data)
        loss = criterion(output.view(-1,ntokens),target)
        #进行反向传播
        loss.backward()
        #先进行梯度规范化,防止梯度消失或者爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
        #进行参数更新
        optimizer.step()
        #损失累加，
        total_loss += loss.item()

        if batch & log_interval  == 0 and batch > 0:
            cur_loss = total_loss/log_interval
            #计算训练到目前的耗时
            elapsed = time.time()-start_time
            #打印日志信息
            print('| epoch{:3d} | {:5d}/{:5d} batches |'
                  '|r {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:5.2f}'.format(epoch,batch,len(train_data)//bptt,scheduler.get_lr()[0],
                                                      elapsed*1000/log_interval,cur_loss,math.exp(cur_loss)))
            total_Loss = 0
            start_time = time.time()


#评估函数的分析
def evaluate(eval_model,data_source):
    eval_model.eval()
    total_loss = 0
    #模型开启评估模式后，不进行反向传播梯度
    with torch.no_grad():
        for i in range(0,data_source.size[0]-1,bptt):
            data,target = get_batch(data_source,i)
            #将原数据放到评估模型进行预测，
            output = eval_model(data)
            #对输出张量进行变形
            output_flat = output.view(-1,ntokens)
            total_loss = criterion(output_flat,target).item()
    return total_loss        

#进行训练和评估
best_val_loss = float("inf")
epochs  = 3
best_model = None
for epoch in range(1,epochs+1):
    start_time = time.time()
    train()
    val_loss = evaluate(model,val_data)
    print('-' * 90)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} |'
          'valid pp; {:8.2f}'.format(epoch,(time.time()-start_time),val_loss,math.exp(val_loss)))
    print('-' * 90)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
    
    scheduler.step()

test_loss = evaluate(best_model,test_data)
print ('-'*90)
print('|End of traning | test loss {:5.2f}'.format(test_loss))
print ('-'*90)



