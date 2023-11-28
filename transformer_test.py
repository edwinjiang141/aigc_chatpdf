import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

# embedding = nn.Embedding(10,3)
# input1 = torch.LongTensor([[1,2,3,4],[5,6,7,8]])
# print(embedding(input1))

##########################  一、构建embedding类实现文本嵌入层 ##########################
class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        # d_model：词嵌入的维度
        # vocab: 词表大小
        super(Embeddings,self).__init__()
        self.lut = nn.Embedding(vocab,d_model)
        self.d_model=d_model
    def forward(self,x):
        #x: 代表输入进模型的文本通过词汇映射后的数字张量
        return self.lut(x)*math.sqrt(self.d_model)

d_model= 512
vocab = 1000
x = Variable(torch.LongTensor([[100,2,421,508],[491,998,1,221]]))
emb = Embeddings(d_model,vocab)
embr = emb(x)
# print("embr:",embr)
# print(embr.shape)

##########################二、构建位置编码器 ##########################
class PostionalEncoding(nn.Module):
    def __init__(self, d_model,dropout,max_len=5000):
        # d_model：词嵌入的维度  dropout: 置0比率  
        super (PostionalEncoding,self).__init__()
        #实例化dropoutcen层
        self.dropout = nn.Dropout(p=dropout)

        #初始化位置矩阵
        pe = torch.zeros(max_len,d_model)
        #初始化绝对位置矩阵
        position = torch.arange(0,max_len).unsqueeze(1)
        #print('postition',position)
        #定义一个变化矩阵
        div_term = torch.exp(torch.arange(0,d_model,2)* - (math.log(10000.0)/d_model))
        #print('div_term',div_term)
        #前面定义的变化举证进行技术、偶数分别赋值
        pe[:,0::2] = torch.sin(position*div_term)
        #print( pe[:,0::2].shape)
        pe[:,1::2] = torch.cos(position*div_term)
        #将二维扩充为三维
        pe = pe.unsqueeze(0)
        #print('fineal pe',pe.shape)
        #注册成buffer，可以在模型保存后重新加载的时候，将这个位置编码器和模型参数一同加载
        self.register_buffer('pe',pe)

    def forward(self,x):
        #文本序列的词嵌入表示    
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False)
        #print('size',x.size(1))
        return self.dropout(x)

d_model = 512
dropout=0.1
max_len = 60

x=embr
pe = PostionalEncoding(d_model,dropout,max_len)
pe_result = pe(embr)
print('pe_result',pe_result)


#构建掩码张量的函数
def subsequent_mask(size):
    attn_shape=(1,size,size)

    #使用np.ones构建全1的张量，在用np.triu形成上三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('unit8')

    #使这个三角矩阵反转
    return torch.from_numpy(1-subsequent_mask)

#注意力机制
# x = Variable(torch.random(5,5))
# mask =  Variable(torch.zeros(5,5))
# y = x.masked_fill(mask==0 ,-1e0)

def attention(query,key,value,mask=None,dropout=None):
    #首先将query的最后一个维度提取出来，代表的是词嵌入的维度
    d_k = query.size(-1)
    #按照注意力计算公式，将query和key的转置进行矩阵乘法，然后除以缩放
    scores = torch.matmul(query,key.transpose(-2,-1))/ math.sqrt(d_k)
    print('scores',scores)
    print('scores shape',scores.shape)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e0)
    #将scores的最后一个维度上进行softmax操作
    p_attn = F.softmax(scores,dim=1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    
    #完成p_attn和value张量的乘法
    return torch.matmul(p_attn,value),p_attn

query = key = value = pe_result
attn,p_attn = attention(query,key,value)
        
print('attn',attn)
print('attn shape',attn.shape)
print('p_attn',p_attn)
print('p_attn shape',p_attn.shape)



# x = torch.randn(4,4)
# # -1 代表自适应。8表示8列
# y = x.view(-1,8)
# print(y)

#多头注意力机制
#实现克隆函数，因为在多头注意一机制下，要用到多个结构相同的线性层
#需要使用clone函数将他们一同初始化到一个网络层列表对象中
def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self,head,embedding_dim,dropout=0.1):
        # head 几个头  embedding_dim  词嵌入维度  dropout dropout的置0的百分比
        super(MultiHeadedAttention,self).__init__()
        # 
        assert embedding_dim & head == 0

        #得到每个头获得的词向量的维度
        self.head = head
        self.d_k = embedding_dim // head
        self.embedding_dim = embedding_dim

        #获得4个线性层，分别是Q,K,V及最终输出的线性层
        self.linars = clones(nn.Linear(embedding_dim,embedding_dim),4)
        #初始化注意力张量
        self.attn = None

        self.dropout = nn.Dropout(p = dropout)
    def forward(self,query,key,value,mask=None):
        #首先判断是否使用掩码张量
        if mask is not None:
            #使用squeeze将掩码张量进行维度扩充，
            mask = mask.unsqueeze(1)

            #得到batch size
            batch_size = query.size(0)

            #首先使用zip将网络层和输入数据连接在一起，模型的输出利用view和transpose进行维度和形状的改变
            query,key,value = \
            [model(x).view(batch_size,-1,self.head,self.d_k).transpose(1,2) 
             for model,x in zip(self.linars,(query,key,value))]
            
            # 将每个头的输出转入到注意力层
            x,self.attn = attention(query,key,value,mask=mask,dropout=self.dropout)

            #得到每个头的计算结果是一个4维张量，需要进行形状的转换
            x = x.transpose(1,2).contiguous().view(batch_size,-1,self.head*self.d_k)
            #最后将x输入线性层列表中最后一个线性层进行处理
            return self.linars[-1](x)
        
head = 8
embedding_dim =512
dropout = 0.2

query = key = value = pe_result
mask = Variable(torch.zeros(2,4,4))

mha = MultiHeadedAttention(head,embedding_dim,dropout)
mha_result = mha(query,key,value,mask)
print('mha_result',mha_result)
print('mha result shape',mha_result.shape)

# 构建前馈全连接网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropput=0.1):
        #d_ff：代表第一个线性层的输出维度，和第二个线性层的输入维度
        super(PositionwiseFeedForward,self).__init__()

        self.w1 = nn.Linear(d_model,d_ff)
        self.w2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self,x):
        # x 代表来自上一层的输出
        #首先将X输入第一个线性层网络，先relu激活，在dropout。在输入到第二个线性层
        return self.w2(self.dropout(F.relu(self.w1(x))))

d_model = 512
d_ff = 64
dropout = 0.2

x = mha_result
ff = PositionwiseFeedForward(d_model,d_ff,dropout)
ff_result = ff(x)
print('ff_result',ff_result)
print('ff_result shape',ff_result.shape)


#规范化层机制
class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        #features 词嵌入的维度
        super(LayerNorm,self).__init__()

        self.a1 = nn.Parameter(torch.ones(features))
        self.b1 = nn.Parameter(torch.zeros(features))

        self.eps = eps
    
    def forward(self,x):
        #分别对X进行最后一个维度上的均值、方差的计算
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)

        return self.a1 * (x-mean)/(std+self.eps)+self.b1
    
features = 512
eps = 1e-6

x = ff_result
ln = LayerNorm(features,eps)
ln_result = ln(x)
print('ln result',ln_result)
print('ln result shape',ln_result.shape)


#子层连接结构
class SublayerConnection(nn.Module):
    def __init__(self,size,dropout=0.1):
        #size 导表词嵌入的维度  dropout的置0比率
        super(SublayerConnection,self).__init__()

        #实例化一个规范化层的对象
        self.norm = LayerNorm(size)
        #实例化一个dropout的对象
        self.dropout = nn.Dropout(p=dropout)
        self.size = size
    
    def forward(self,x,sublayer):
        #sublayer 该子层连接中子层函数
        #首先将x进行规范化，然后送入子层函数中，处理结果进去dropout层，最后进行残差连接（跳跃来接）
        return x+self.dropout(sublayer(self.norm(x)))

size = d_model = 512
head = 4
dropout = 0.2
x = pe_result
mask = Variable(torch.zeros(2,4,4))
self_attn = MultiHeadedAttention(head,d_model)
sublayer = lambda x: self_attn(x,x,x,mask)

sc = SublayerConnection(size,dropout)
sc_result = sc(x,sublayer)
print('sc result',sc_result)
print('sc result shape',sc_result.shape)



#编码器层实现
class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):

        super(EncoderLayer,self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size

        #编码器层中有2个子层连接，使用clones函数进行操作
        self.sublayer = clones(SublayerConnection(size,dropout),2)
    
    def forward(self,x,mask):
        # x 代表上一层的张量  mask 掩码张量
        #首先让x经过第一子层连接结构，内部包含多头注意力子层
        #再让张量经过第二子层连接结构，其中包含前馈全连接网络
        x = self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)

size = d_model = 512
head = 4
d_ff = 64
x = pe_result
dropout = 0.2

self_attn = MultiHeadedAttention(head,d_model)
ff = PositionwiseFeedForward(d_model,d_ff,dropout)
mask = Variable(torch.zeros(2,4,4))

el = EncoderLayer(size,self_attn,ff,dropout)
el_result = el(x,mask)

print('el result',el_result)
print('el result shape',el_result.shape)


#解码器层实现   每个解码器层根据给定的输入向目标方向进行特征提取操作
class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        #self_attn  多头自注意力机制对象
        #src_attn   常规的注意力机制的对象
        #feed_forward 前馈全连接层对象
        super(DecoderLayer,self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout

        #按照解码器层的结构，用clone函数克隆3个子层连接对象
        self.sublayer = clones(SublayerConnection(size,dropout),3)
        
    def forward(self,x,memory,source_mask,target_mask):
        #x 代表上一层输入的张量
        # memory：代表编码器的语义存储张量
        m = memory
        #第一步让x经历第一个子层，多头自注意力机制的子层
        #采用target_mask, 为了将解码时未来的信息进行过滤，比如模型解码第二个字符
        x = self.sublayer[0](x,lambda x: self.self_attn(x,x,x,target_mask))

        #第二步让x经历第二个子层，常规的注意力机制的子层 Q!=K=V
        #采用source_mask，为了遮蔽对信息无用的数据
        x = self.sublayer[1](x,lambda x: self.src_attn(x,m,m,source_mask))

        #第三步让x经历第三个子层，前馈全连接层
        return self.sublayer[2](x,self.feed_forward)

size = d_model = 512
head = 4
d_ff = 64
dropout = 0.2
self_attn = src_attn = MultiHeadedAttention(head,d_model,dropout)

ff = PositionwiseFeedForward(d_model,d_ff,dropout)

x = pe_result

memory = el_result

mask = Variable(torch.zeros(2,4,4))

source_mask = target_mask = mask

dl = DecoderLayer(size,self_attn,src_attn,ff,dropout)

dl_result = dl(x,memory,source_mask,target_mask)

print('dl reslt',dl_result)
print('dl result shape',dl_result.shape)


#解码器实现
class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()

        #clones函数克隆N个layer
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)

    def forward(self,x,memory,source_mask,target_mask):
        #x 代表上一层输入的张量
        # memory：代表编码器的语义存储张量
        #将X 依次经历所有的编码器层处理，最后经过规范化层
        for layer in self.layers:
            x = layer(x,memory,source_mask,target_mask)
        return self.norm(x)

size = d_model = 512
head =4
d_ff = 64
dropout = 0.2
c = copy.deepcopy

attn = MultiHeadedAttention(head,d_model)
ff = PositionwiseFeedForward(d_model,d_ff,dropout)

layer = DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout)
N=8
x = pe_result
memory = el_result

mask = Variable(torch.zeros(2,4,4))
source_mask = target_mask = mask

de = Decoder(layer,N)
de_result = de(x,memory,source_mask,target_mask)

print('de reslt',de_result)
print('de result shape',de_result.shape)


#输出部分实现
import torch.nn.functional as F
class Generator(nn.Module):
    def __init__(self,d_model,vocab_size):
        super(Generator,self).__init__()
        #定义一个线性层，作用是弯沉网路输出维度的变换
        self.project = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return F.log_softmax(self.project(x),dim=-1)

d_mpdel = 512
vocab_siez = 1000
x = de_result

gen = Generator(d_model,vocab_siez)
gen_result = gen(x)


print('gen reslt',gen_result)
print('gen result shape',gen_result.shape)

#模型的构建
