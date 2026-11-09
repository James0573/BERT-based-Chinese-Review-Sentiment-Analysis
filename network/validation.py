from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset_process import MyDataset
from net import Model, pretrained


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_DIR = Path(__file__).resolve().parent / "trained_model"

tokenizer = BertTokenizer.from_pretrained(
    "/Users/guoyanheng/Documents/code/shop_comments_train/model/models--google-bert--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea"
)


def collate_fn(data):
    sentences = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 与 trainer、tester 保持相同的分词、截断和补齐方式
    data = tokenizer(
        text=sentences,
        truncation=True,
        max_length=350,
        return_tensors="pt",
        padding="max_length",
    )

    return (
        data["input_ids"],
        data["attention_mask"],
        data["token_type_ids"],
        torch.LongTensor(labels),
    )

validation_dataset = MyDataset("validation")
validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

if __name__ == "__main__":
    # 只比较 trainer 保存的数字编号文件，按 epoch 数值排序
    model_files = sorted(
        (
            path for path in MODEL_DIR.glob("*BertModel.pt")# 只选择以"BertModel.pt"结尾的文件
            if path.stem.removesuffix("BertModel").isdigit()# 只选择文件名去掉"BertModel"后是数字的文件
        ),
        key=lambda path: int(path.stem.removesuffix("BertModel")),  #按文件名去掉"BertModel"后转换为整数进行排序
    )



    model = Model().to(DEVICE)
    model.eval()
    pretrained.eval()  # BERT 是 net.py 中的独立对象，需要单独设置评估模式
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")  #计算一个batch的总损失，方便后续计算平均损失

    best_file = None 
    best_accuracy = -1.0 #-1.0表示没有模型被评估过
    best_loss = float("inf")

    print(f"Using device: {DEVICE}")
    print(f"验证集：{len(validation_dataset)} 条，共比较 {len(model_files)} 个模型")

    with torch.no_grad():
        for model_file in model_files:
            model.load_state_dict(torch.load(model_file, map_location=DEVICE, weights_only=True))
            
            all_correct_number = 0
            total = 0
            total_loss = 0.0

            for input_ids, attention_mask, token_type_ids, labels in validation_dataloader:
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                token_type_ids = token_type_ids.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(input_ids, attention_mask, token_type_ids) #logits
                total_loss += loss_fn(outputs, labels).item() #为什么加和loss
                predictions = outputs.argmax(dim=1) #将logits分数转换为label标签
                all_correct_number += (predictions == labels).sum().item() #正确预测数
                total += labels.size(0) #样本总数

            accuracy = all_correct_number / total #准确率
            average_loss = total_loss / total #平均损失
            epoch = int(model_file.stem.removesuffix("BertModel")) #epoch编号
            print(f"epoch={epoch} | Validation accuracy={accuracy:.4f} | loss={average_loss:.4f}")

            # 准确率优先；准确率相同时，选择平均损失更小的模型
            if accuracy > best_accuracy or (accuracy == best_accuracy and average_loss < best_loss):
                best_file = model_file
                best_accuracy = accuracy
                best_loss = average_loss

    best_epoch = int(best_file.stem.removesuffix("BertModel"))
    print(f"\n最佳 epoch={best_epoch}（编号从 0 开始）",f"Validation accuracy={best_accuracy:.4f} | loss={best_loss:.4f}",f"最佳模型文件：{best_file}")
