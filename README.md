# BERT-based-Chinese-Review-Sentiment-Analysis
基于预训练 BERT 与可训练分类头，实现中文评论正负面分类，支持批量训练、验证集选模、测试评估与交互式预测。

预训练中文模型（建立一个model文件夹后自行下载）：[models--google-bert--bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)

数据集：/dataset/lansinuote___chn_senti_corp/default/0.0.0/b0c4c119c3fb33b8e735969202ef9ad13d717e5a

数据集处理：/network/dataset_process.py

微调下游分类模型搭建：/network/net.py

训练：/network/trainer.py

评估：/network/validation.oy

测试：/network/tester.py

评论输入交互窗口：test_command.py
