'''
[中文命名实体识别]基于HFL/RBT6的模型微调的中文命名实体识别项目
预训练模型:(huggingface)HFL/RBT6,哈工大的基于RoBERTa的中文训练模型,见https://huggingface.co/hfl/rbt6.
下游模型:增加一层GRU和一层全连接层.
训练方式:两段式训练,先把下游模型中的参数大致训练1次,然后带着预训练模型一起训练2次.
数据集:人民日报的训练集,见https://huggingface.co/datasets/peoples_daily_ner.
'''
# -加载编码工具
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('hfl/rbt6') 
# print(tokenizer)

# -定义数据集
import torch
from datasets import load_dataset, load_from_disk
class Dataset(torch.utils.data.Dataset):
    def __init__(self,split):
        path = 'E:/vscode_doc/data/NER_in_Chinese'
        dataset = load_from_disk(dataset_path = path)[split]
        def f(data):
            return len(data['tokens']) <= 512-2 # 过滤太长的句子
        dataset = dataset.filter(f)
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,i):
        tokens = self.dataset[i]['tokens']
        labels = self.dataset[i]['ner_tags']
        return tokens, labels

dataset = Dataset('train')
# tokens, labels = dataset[0]
# print(len(dataset), tokens, labels)

# labels中的names = ['0','B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC']
# names[i],i=0,1,2,3,4,5,6分别表示
# 非实体,人名(开始词),人名(中间词),组织(开始词),组织(中间词),地点(开始词),地点(中间词)

# -数据整理函数
def collate_fn(data):
    tokens = [i[0] for i in data]
    labels = [i[1] for i in data]
    inputs = tokenizer.batch_encode_plus(
        tokens,
        truncation=True,
        padding=True,
        return_tensors='pt', 
        is_split_into_words=True,
    )
    lens = inputs['input_ids'].shape[1] # 找出本批次中最长的句子的长度
    for i in range(len(labels)):
        labels[i] = [7] + labels[i]     # 头部加一个[7],这里对应[CLS]
        labels[i] += [7] * lens         # 尾部也加一个[7],这里对应[SEP]和[PAD]
        labels[i] = labels[i][:lens]    # 所有labels都统一到lens长度
    
    return inputs, torch.LongTensor(labels)

# -数据加载器
loader = torch.utils.data.DataLoader(
    dataset = dataset,
    batch_size = 16,
    collate_fn = collate_fn,
    shuffle = True,
    drop_last = True
)

for i,(inputs, labels) in enumerate(loader):
      break
# print(len(loader))
# print(tokenizer.decode(inputs['input_ids'][0]))
# print(labels[0])

# for k,v in inputs.items():
#       print(k, v.shape)

# inputs包括了input_ids, token_type_ids, attention_mask

# -加载预训练模型
from transformers import AutoModel
pretrained = AutoModel.from_pretrained('hfl/rbt6') # 用的是哈工大的模型
# print(sum(i.numel() for i in pretrained.parameters())/10000)
# 存放路径：C:\Users\Administrator\.cache\huggingface\hub

# print(pretrained(**inputs).last_hidden_state.shape)

# -定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tuneing = False
        self.pretrained = None
        
        self.rnn = torch.nn.GRU(768,768)
        self.fc = torch.nn.Linear(768,8)
        # 下游任务模型只有一个GRU和一个全连接层

    def forward(self, inputs):
        # 如果tuneing=True,则与预训练模型属于模型一部分,用预训练模型计算参数梯度
        # 否则不属于自己模型一部分,不计算参数梯度
        if self.tuneing:
            out = self.pretrained(**inputs).last_hidden_state
        else:
            with torch.no_grad():
                out = pretrained(**inputs).last_hidden_state
        # **inputs表示把传入的inputs参数储存为字典

        out, _ = self.rnn(out)
        out = self.fc(out).softmax(dim=2)
        return out
    
    def fine_tuneing(self,tuneing):
        self.tuneing = tuneing
        if tuneing:
            for i in pretrained.parameters():
                i.requires_grad = True
            pretrained.train() # 处于tuneing模式则开启dropout等功能
            self.pretrained = pretrained
        else:
            for i in pretrained.parameters():
                i.requires_grad_(False)
            pretrained.eval()
            self.pretrained = None

model = Model()
# print(model(inputs).shape)

# -定义2个工具函数
def reshape_and_remove_pad(outs, labels, attention_mask):
    '''
    主要是用来调整输出和可视化的
    '''
    # 变形用于计算loss
    outs = outs.reshape(-1,8)   # [b,lens,8] -> [b*lens,8]
    labels = labels.reshape(-1) # [b,lens] -> [b*lens]

    select = attention_mask.reshape(-1) == 1 
    # attention_mask中=1的就不是pad,0都是pad
    # 移除pad,否则会影响计算正确率
    outs = outs[select]
    labels = labels[select]

    return outs, labels

# print(reshape_and_remove_pad(torch.randn(2,3,8),torch.ones(2,3),torch.ones(2,3)))

def get_correct_and_total_count(labels, outs):
    outs = outs.argmax(dim=1) # [b*lens,8] -> [b*lens]
    correct = (outs == labels).sum().item()
    total = len(labels)

    # labels=0的情况太多(即句子中没有实体的部分太多),要排除掉labels=0后再计算正确率
    # 这里计算了2组正确率,一组是含0的correct,total,另一组是排除掉0的
    select = labels != 0 
    outs = outs[select]
    labels = labels[select]
    correct_content = (outs==labels).sum().item()
    total_content = len(labels)
    return correct, total, correct_content, total_content

# print(get_correct_and_total_count(torch.ones(16),torch.randn(16,8)))

# -训练
def train(epochs):
    lr = 2e-5 if model.tuneing else 5e-4

    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        for step, (inputs,labels) in enumerate(loader):
            outs = model(inputs) # [b,lens]->[b,lens,8]
            outs, labels = reshape_and_remove_pad(
                outs,labels,inputs['attention_mask']
            )
            # outs: [b,lens,8]->[c,8]
            # labels: [b,lens]->[c]

            #梯度下降
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step %50 == 0:
                counts = get_correct_and_total_count(labels,outs)
                acc = counts[0] / counts[1]
                acc_content = counts[2] / counts[3]
                print(epoch,step,loss.item(),acc, acc_content)
        torch.save(model, '/vscode_doc/model/NER_ch_01.model')

# model.fine_tuneing(False)
# train(1)
# model.fine_tuneing(True)
# train(2)
# 两段式训练,先把下游模型中的参数大致训练1次,然后带着预训练模型一起训练2次

# -测试
'''
限于本地cpu能力,训练好后模型保存于本地.这里不再重新训练,直接调取训练好的模型.
'''
def test():
    model_load = torch.load('/vscode_doc/model/命名实体识别_中文.model')
    model_load.eval()
    loader_test = torch.utils.data.DataLoader(
        dataset = Dataset('validation'),
        batch_size = 128,
        collate_fn = collate_fn,
        shuffle = True,
        drop_last = True
    )
    correct, total = 0, 0
    correct_content, total_content = 0, 0

    for step,(inputs,labels) in enumerate(loader_test):
        if step == 10:
            break
        with torch.no_grad():
            # [b,lens]->[b,lens,8]->[b,lens]
            outs = model_load(inputs)
            outs, labels = reshape_and_remove_pad(
                outs,labels,inputs['attention_mask']
            )
            counts = get_correct_and_total_count(labels,outs)
            correct += counts[0]
            total += counts[1]
            correct_content += counts[2]
            total_content += counts[3]
        print(step,correct/total, correct_content/total_content)

# test()

# -预测,可视化
def predict():
    model_load = torch.load('/vscode_doc/model/命名实体识别_中文.model')
    model_load.eval()
    loader_test = torch.utils.data.DataLoader(
        dataset = Dataset('validation'),
        batch_size = 32,
        collate_fn = collate_fn,
        shuffle = True,
        drop_last = True
    )

    for i,(inputs,labels) in enumerate(loader_test):
        break
    with torch.no_grad():
        outs = model_load(inputs).argmax(dim=2)
    
    for i in range(32):
        select = inputs['attention_mask'][i] == 1
        input_id = inputs['input_ids'][i,select]
        out = outs[i,select]
        label = labels[i,select]
        # 输出原句子
        print(tokenizer.decode(input_id).replace(' ',''))
        # 输出tag
        for tag in [label,out]:
            s = ''
            for j in range(len(tag)):
                if tag[j] == 0:
                    s += '·'
                    continue
                s += tokenizer.decode(input_id[j])
                s += str(tag[j].item())
            print(s)
        print('============================')
        
predict()