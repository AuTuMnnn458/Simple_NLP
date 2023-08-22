# Simple_NLP
## 一些NLP实战项目
本项目包括一些简单的NLP实战的项目，目前包括:
1. 【文本分类/情感分类】基于Distilbert模型微调的对Cola数据集的情感分类项目
2. 【中文命名实体识别】基于HFL/RBT6的模型微调的中文命名实体识别项目
3. 【文本摘要】

以上传.py文件为主要形式，项目基本都是基于大模型的微调。限于本地算力和实现可视化的困难，部分项目的训练过程以及可视化不会进行。

## 项目简要介绍
### 1.【文本分类/情感分类】基于Distilbert模型微调的对Cola数据集的情感分类项目
* 数据集: glue中的cola,见https://huggingface.co/datasets/glue
* 预训练模型: (huggingface)distilbert-base-uncased,见https://huggingface.co/distilbert-base-uncased
* 下游任务模型: 再加一个fc + ReLU + Dropout + fc

### 2.【中文命名实体识别】基于HFL/RBT6的模型微调的中文命名实体识别项目
* 预训练模型:(huggingface)HFL/RBT6,哈工大的基于RoBERTa的中文训练模型,见https://huggingface.co/hfl/rbt6.
* 下游模型:增加一层GRU和一层全连接层.
* 训练方式:两段式训练,先把下游模型中的参数大致训练1次,然后带着预训练模型一起训练2次.
* 数据集:人民日报的训练集,见https://huggingface.co/datasets/peoples_daily_ner.

### 3.【文本摘要】
