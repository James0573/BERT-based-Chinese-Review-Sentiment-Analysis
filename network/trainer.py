import torch #PyTorch库
from dataset_process import MyDataset #加载自定义数据集类MyDataset，用于处理和加载训练数据
from torch.utils.data import DataLoader #PyTorch的数据加载器DataLoader，用于批量加载数据
from net import Model #加载下游预训练网络
from transformers import BertTokenizer #加载BERT分词器
from torch.optim import AdamW #加载AdamW优化器

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 100 #训练轮数

tokenizer = BertTokenizer.from_pretrained(r"/Users/guoyanheng/Documents/code/shop_comments_train/model/models--google-bert--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea") #加载BERT分词器，用于将文本转换为模型可接受的输入格式


#定义函数对数据编码
def collate_fn(data):
    #将文本和标签分开
    sentences =[i[0] for i in data] #提取数据中的句子
    labels = [i[1] for i in data] #提取数据中的标签

    #调用tokenizer对文本进行编码
    data=tokenizer(
        text=sentences, #将句子列表传入分词器进行批量编码
        truncation=True, #截断文本，超过最大长度的部分将被截断, 
        max_length=350, #最大长度为350个token,
        return_tensors='pt' ,# 返回PyTorch张量, 
        padding='max_length', #补齐到最大长度，短于最大长度的部分将被填充
        return_length=True # 返回编码后的文本长度
    )
    # 向量化，包括input_ids、attention_mask和token_type_ids，以及将标签转换为张量形式。
    input_ids=data["input_ids"] #获取编码后的输入ID
    attention_mask=data["attention_mask"] #获取注意力掩码
    token_type_ids=data["token_type_ids"] #获取token类型ID

    #处理标签，向量化为张量形式
    labels=torch.LongTensor(labels) #将标签转换为张量

    return input_ids,attention_mask,token_type_ids,labels #返回处理后的数据



#创建数据集，调用MyDataset类，传入训练集的划分参数"train"
train_dataset = MyDataset("train") 

#创建数据加载器
train_dataloader = DataLoader(
    dataset=train_dataset, #训练数据集
    batch_size=32, #一次加载32条数据
    shuffle=True,#打乱数据
    drop_last=True,#最后一个batch如果不够32条数据，则丢弃
    collate_fn=collate_fn #采用自定义的collate_fn函数，对每个batch的数据进行编码和处理
    ) 

if __name__ == "__main__":
    # 在这里可以添加训练循环或其他主程序逻辑
    print(f"Using device: {DEVICE}") #打印当前使用的设备
    
    model = Model().to(DEVICE) #将模型移动到指定设备
    optimizer = AdamW(model.fc.parameters(), lr=1e-4) #定义优化器，使用AdamW优化器，学习率为1e-4,且只优化全连接层（fc）的参数
    loss_fn = torch.nn.CrossEntropyLoss("mean") #定义损失函数，已经包含softmax操作，希望接收未归一化的logits作为输入,mean表示返回一个batch的平均损失值
    model.train() #将模型设置为训练模式
    
    print(f"Starting training for {EPOCHS} epochs") #打印训练开始信息


    for epoch in range(EPOCHS):#遍历每个训练轮数
        
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_dataloader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)#前向传播计算输出
            loss = loss_fn(outputs, labels) #计算损失，将未归一化的logits传入损失函数，损失函数内部会进行softmax操作，得到每个类别的概率分布
            
            optimizer.zero_grad() #清空梯度
            loss.backward() #反向传播计算梯度
            optimizer.step() #更新模型参数

            if i%5 == 0: #每5个batch打印一次损失
            
                outputs = outputs.argmax(dim=1) #将类别概率转换为预测的类别索引

                accuracy = (outputs == labels).float().mean().item() #计算当前batch的准确率
                print(i,epoch,accuracy,loss.item()) #打印当前batch的索引、当前轮数、准确率和损失

        torch.save(model.state_dict(), f"/Users/guoyanheng/Documents/code/shop_comments_train/network/trained_model/{epoch}BertModel.pt") #保存训练后的模型参数
        print(f"Saved model for epoch {epoch}") #打印模型保存信息
