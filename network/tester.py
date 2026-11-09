
import torch #PyTorch库
from dataset_process import MyDataset #加载自定义数据集类MyDataset，用于处理和加载训练数据
from torch.utils.data import DataLoader #PyTorch的数据加载器DataLoader，用于批量加载数据
from net import Model #加载下游模型
from transformers import BertTokenizer #加载BERT分词器


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #设置设备，优先使用MPS，否则使用CPU

tokenizer = BertTokenizer.from_pretrained(r"/Users/guoyanheng/Documents/code/shop_comments_train/model/models--google-bert--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea") #加载BERT分词器，用于将文本转换为模型可接受的输入格式

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
    )
    # 向量化，包括input_ids、attention_mask和token_type_ids，以及将标签转换为张量形式。
    input_ids=data["input_ids"] #获取编码后的输入ID
    attention_mask=data["attention_mask"] #获取注意力掩码
    token_type_ids=data["token_type_ids"] #获取token类型ID

    #处理标签，向量化为张量形式
    labels=torch.LongTensor(labels) #将标签转换为张量

    return input_ids,attention_mask,token_type_ids,labels #返回处理后的数据

test_dataset = MyDataset("test") #创建测试集
test_dataloader = DataLoader(
    dataset=test_dataset, #测试数据集
    batch_size=32, #一次加载32条数据
    shuffle=False, #测试集不打乱
    drop_last=False, #最后一个batch如果不够32条数据，则保留
    collate_fn=collate_fn #采用自定义的collate_fn函数，对每个batch的数据进行编码和处理
)

if __name__ == "__main__":
    # 在这里可以添加测试循环或其他主程序逻辑
    all_correct_number = 0 #初始化准确样本数
    total = 0 #初始化总样本数
    print(f"Using device: {DEVICE}") #打印当前使用的设备
    
    model = Model().to(DEVICE) #将模型移动到指定设备
    model.load_state_dict(torch.load("/Users/guoyanheng/Documents/code/shop_comments_train/network/trained_model/2BertModel.pt", map_location=DEVICE)) #加载训练好的模型权重
    model.eval() #将模型设置为评估模式 

    with torch.no_grad(): #在测试阶段，不需要计算梯度
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_dataloader):
            
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(input_ids, attention_mask, token_type_ids)#获取模型的原始输出，每个样本的logits
                #outputs = outputs.softmax(dim=1) #softmax归一化，得概率分布
            outputs = outputs.argmax(dim=1) #将logits分数转换为label标签

            #每一轮的准确率计算
            batch_correct_number = (outputs == labels).float().sum().item() #计算当前batch的准确样本数,即和label一致的样本数
            batch_total = labels.size(0) #当前batch的样本总数
            print(f"Batch {i + 1}: 当前batch准确率={batch_correct_number / batch_total:.4f}")
            
            #循环一轮累加一轮
            all_correct_number += batch_correct_number #累加当前batch的准确样本数
            total += batch_total #累加当前batch的样本数

            

    accuracy = all_correct_number / total #计算整体准确率
    print(f"Test accuracy: {accuracy:.4f}") #打印整体准确率