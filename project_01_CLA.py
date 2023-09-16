'''
[文本分类/情感分类]基于Distilbert模型微调的对Cola数据集的情感分类项目
数据集: glue中的cola,见https://huggingface.co/datasets/glue
预训练模型: (huggingface)distilbert-base-uncased,见https://huggingface.co/distilbert-base-uncased
下游任务模型: 再加一个fc + ReLU + Dropout + fc 
'''

# -编码器
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast = True)
# print(tokenizer)

# -数据集
from datasets import load_from_disk
dataset = load_from_disk('/vscode_doc/data/glue/cola')

def f(a):
    return tokenizer.batch_encode_plus(
        a['sentence'], truncation=True
    ) # 分词
dataset = dataset.map(
    function=f,batched=True,batch_size=1000,
    remove_columns=['sentence','idx'] # 删除多余字段
)

# print(dataset['train'][0])
# print(dataset)

# -数据集加载器
import torch
from transformers.data.data_collator import DataCollatorWithPadding
# huggingface提供的DataCollatorWithPadding,对数据自动补充padding
loader = torch.utils.data.DataLoader(
    dataset = dataset['train'], batch_size=8, 
    collate_fn = DataCollatorWithPadding(tokenizer),
    shuffle=True, drop_last=True
)
# for i, data in enumerate(loader):
#    break
# for k,v in data.items():
#    print(k,v.shape,v[:3])
# print(len(loader))


# -构建模型
from transformers import AutoModelForSequenceClassification, DistilBertModel
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = DistilBertModel.from_pretrained(
            'distilbert-base-uncased')
        # 下游任务模型
        # 只用了非常简单的线性网络
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768,768),
            torch.nn.ReLU(), torch.nn.Dropout(p=0.2),
            torch.nn.Linear(768,2)
        ) 

        # 加载预训练模型的参数,不是必要的
        parameters = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels = 2)
        self.fc[0].load_state_dict(parameters.pre_classifier.state_dict())
        self.fc[3].load_state_dict(parameters.classifier.state_dict())
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.pretrained(input_ids=input_ids,
                                 attention_mask = attention_mask)
        logits = logits.last_hidden_state[:,0]
        logits = self.fc(logits) 

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return {'loss':loss, 'logits':logits}

model = Model()
# print(sum(i.numel() for i in model.parameters())/10000)

# out = model(**data)
# print(out['loss'],out['logits'].shape)

# -加载评价指标(cola专用)
# 网络原因这里不考虑
# from datasets import load_metric
# metric = load_metric(path='glue', config_name = 'cola')
# print(metric.compute(predictions=[0,1,1,0],references=[0,1,1,1]))

# -测试
def test():
    model.eval()

    load_test = torch.utils.data.DataLoader(
        dataset = dataset['validation'],
        batch_size=16,
        collate_fn = DataCollatorWithPadding(tokenizer),
        shuffle=True,
        drop_last=True
    )

    outs= []
    labels=[]
    for i, data in enumerate(load_test):
        with torch.no_grad():
            out = model(**data)
        
        outs.append(out['logits'].argmax(dim=1))
        labels.append(data['labels'])
        if i % 10 == 0:
            out_p = torch.cat(outs)
            label_p = torch.cat(labels)
            acc = (out_p==label_p).sum().item() / len(label_p)
            print(i,acc)
        if i ==50:
            break
# print(test())

'''
out_p = torch.cat(outs)
label_p = torch.cat(labels)
acc = (out_p==label_p).sum().item() / len(label_p)
'''

# -训练函数
from transformers.optimization import get_scheduler

def train():
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5)
    scheduler = get_scheduler(
        name = 'linear',num_warmup_steps=0,num_training_steps=len(loader),optimizer=optimizer
    )

    model.train()
    for i,data in enumerate(loader):
        out = model(**data)
        loss = out['loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad()
        model.zero_grad()

        if i % 50 == 0:
            out = out['logits'].argmax(dim=1)
            acc = (data['labels'] == out).sum().item() / 8
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(i, loss.item(), acc, lr)           

    torch.save(model,'/vscode_doc/model/project_01_CLA.model')

# train()

# -可视化
import numpy as np
import pandas as pd
# 训练得到的数据已经保存在本地
df = pd.read_csv('/vscode_doc/doc/project_01_CLA_result.txt',header = None,sep=' ') 
df = pd.DataFrame(df)
df.columns = ['i','loss','acc','lr']
# print(data)
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('seaborn')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

plt.plot(df['i'],df['loss'],label = 'loss')
plt.plot(df['i'],df['acc'],label = 'accuracy')
plt.title('训练次数-损失函数/准确率图像')
plt.legend()
# plt.show()

plt.plot(df['i'],df['lr'],label = 'learning rate')
# plt.show()