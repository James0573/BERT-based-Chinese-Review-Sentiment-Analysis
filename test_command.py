import torch
from net import Model
from transformers import BertTokenizer


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(
    r"/Users/guoyanheng/Documents/code/shop_comments_train/model/models--google-bert--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea")

def  data_loader(sentence):
    data = tokenizer(
            text=[sentence],
            truncation=True,
            max_length=350,
            return_tensors="pt",
            padding="max_length",
        )
    
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    
    return input_ids, attention_mask, token_type_ids


if __name__ == "__main__":

    model = Model().to(DEVICE)
    model.load_state_dict(torch.load("/Users/guoyanheng/Documents/code/shop_comments_train/network/trained_model/2BertModel.pt",map_location=DEVICE))
    model.eval()

    while True:
        text = input("请输入评论（输入 q 退出）：").strip()

        if text.lower() == "q":
            break
        if not text:
            print("评论不能为空，请重新输入。")
            continue

        input_ids, attention_mask, token_type_ids = data_loader(text)

        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
            label = outputs.argmax(dim=1).item()

        sentiment = "optimistic" if label == 1 else "negative"
        print(f"标签：{label}，情感：{sentiment}")
