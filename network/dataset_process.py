#from pathlib import Path

from torch.utils.data import Dataset
from datasets import load_dataset


class MyDataset(Dataset):

    def __init__(self, split):
        #通过huggingface缓存路径来读取数据集
        # self.dataset = load_dataset(
        #     path="lansinuote/ChnSentiCorp",
        #     split= split,
        #     cache_dir="/Users/guoyanheng/Documents/code/shop_comments_train/dataset")
            
        #如果加载本地文件，写成:
        self.dataset = load_dataset(
                    path="arrow", #arrow是huggingface数据集的格式之一，适合大规模数据集的存储和读取
                    split= split,
                    data_files={"train": "/Users/guoyanheng/Documents/code/shop_comments_train/dataset/lansinuote___chn_senti_corp/default/0.0.0/b0c4c119c3fb33b8e735969202ef9ad13d717e5a/chn_senti_corp-train.arrow",
                "test": "/Users/guoyanheng/Documents/code/shop_comments_train/dataset/lansinuote___chn_senti_corp/default/0.0.0/b0c4c119c3fb33b8e735969202ef9ad13d717e5a/chn_senti_corp-test.arrow",
                "validation":"/Users/guoyanheng/Documents/code/shop_comments_train/dataset/lansinuote___chn_senti_corp/default/0.0.0/b0c4c119c3fb33b8e735969202ef9ad13d717e5a/chn_senti_corp-validation.arrow"
                })

    def __len__(self): # 返回数据集的样本数
        return len(self.dataset)

    def __getitem__(self, item): # 返回指定索引的样本
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]
        return text,label
pass


if __name__ == "__main__":
    dataset = MyDataset("validation")
    print(f"样本数={len(dataset)}")
    print("首条样本:", dataset[:20])