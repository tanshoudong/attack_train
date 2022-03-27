# Attack-Train-Compare-Pytorch

在中文文本分类的场景下，以TextCNN为基准模型,综合对比不同对抗训练方法(FGSM、FGM、PGD、FreeAT)的效果，基于Pytorch实现。

## 中文数据集
我从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，已上传至github，文本长度在20到30之间。一共10个类别，每类2万条。
类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。


## 运行环境
ubantu 16.04  
cuda 10.2  
cudnn 8.2  
NVIDIA Tesla P40 显卡 24G显存  
python 3.6  
pytorch 1.6  

## 实验参数
batch_size 1024  
embedding_dim 128  
训练集 18w  
验证集 1w  
测试集 1w  

## 特别说明
1.本次实验数据以字为单位输入模型，没有使用预训练词向量，初始化向量维度为128
2.由于预训练词向量维度300，训练较慢，故没有采用，且从最终实验效果看，用不用预训练词向量，指标效果差别不大
3.读者若想使用预训练词向量，可自行配置。

预训练词向量 [搜狗新闻 Word+Character 300d](https://github.com/Embedding/Chinese-Word-Vectors)，[点这里下载](https://pan.baidu.com/s/14k-9jsspp43ZhMxqPmsWMQ) 


## 测试集效果

baseline+attack_train|precision|recall|macro_F1
--|--|--|--
TextCNN|90.63%|90.62%|90.62%s
TextCNN+FGSM|91.67%|91.65%|91.65%
TextCNN+FGM|90.96%|90.93%|90.94%
TextCNN+PGD|90.82%|90.80%|90.80%
TextCNN+FreeAT|90.65%|90.62%|90.62%

## 使用说明
```
# 训练并测试：
# TextCNN
python3 run.py --attack_train ""

# TextCNN + FGSM
python3 run.py --attack_train "fgsm"

# TextCNN + FGM
python3 run.py --attack_train "fgm"

# TextCNN + PGD
python3 run.py --attack_train "pgd"

# TextCNN + FreeAT
python3 run.py --attack_train "FreeAT"

```

备注：
    1.依次执行上述命令，便可复现测试集的效果；
    2.对于每种对抗训练方法，还可以精细调节eps参数，效果应该还可以提升，上述实验只尝试了一种eps参数值。
