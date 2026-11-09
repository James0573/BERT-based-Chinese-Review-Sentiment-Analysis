#上游为预训练模型
#预训练模型输出，传递给下游微调模型
from turtle import forward 

import torch 
from transformers import AutoModel

#from trainer import DEVICE

#配置GPU优先
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(DEVICE) 

#加载预训练模型
pretrained = AutoModel.from_pretrained("/Users/guoyanheng/Documents/code/shop_comments_train/model/models--google-bert--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea").to(DEVICE)
print(pretrained)

#定义下游任务模型（将主干网络提取特征并分类）
class Model(torch.nn.Module):#继承自torch.nn.Module的类，表示一个神经网络模型
    
    def __init__(self): #类的构造函数，用于初始化模型的参数和结构
        super().__init__() #调用父类的构造函数，初始化模型的基本属性
        self.fc=torch.nn.Linear(768, 2) #定义一个全连接层，输入维度为768(预训练模型的输出维度)  ，输出维度为2（类别数），用于分类任务
    
    def forward(self,input_ids,attention_mask,token_type_ids): #模型的前向传播函数，用于定义输入数据如何经过模型计算得到输出结果
        
        #上游任务BERT不参与训练，冻结BERT的参数，只用于提取特征
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) 
        #下游任务二分类模型参与训练
        out=self.fc(out.last_hidden_state[:,0]) #只取上游模型的最后一层的[CLS]标记的输出作为特征向量，传入全连接层进行分类
        #out=out.softmax(dim=1) #对输出进行softmax归一化，得到每个类别的概率分布
        return out #返回全连接层的输出，即未归一化的logits


