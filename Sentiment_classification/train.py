#coding=utf-8
from load_data import make_loader
from transformers import BertTokenizer
from transformers import BertModel
import torch
from torch.optim import AdamW


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0])

        out = out.softmax(dim=1)

        return out






def collate_fn(data,labels):
    #编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=data,
                                   #当句子大于max_length时截断
                                   truncation=True,
                                   # 当句子小于max_length时填充
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   #token_type_ids第一个句子和特殊符号的位置时0，第二个句子位置是1
                                   return_token_type_ids=True,
                                   #padding的位置是0，其它位置是1
                                   return_attention_mask=True,
                                   return_length=True)

    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('run on:',device)
#获得训练，验证和测试的批数据
train_loader,val_loader,test_loader = make_loader()
#加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')
#加载预训练模型
pretrained = BertModel.from_pretrained('bert-base-chinese')
pretrained = pretrained.to(device)
#初始化下游任务模型
model = Model()
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()
def train(epochs):
    for i in range(epochs):
        model.train()
        for i,data in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels = collate_fn(data['text'],data['label'])
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
            out = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 5 == 0:
                out = out.argmax(dim=1)
                accuracy = (out == labels).sum().item() / len(labels)

                print('train:',i, loss.item(), accuracy)
        model.eval()
        val_acc = []
        for i, data in enumerate(val_loader):
            input_ids, attention_mask, token_type_ids, labels = collate_fn(data['text'], data['label'])
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device), labels.to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            if i % 5 == 0:
                out = out.argmax(dim=1)
                accuracy = (out == labels).sum().item() / len(labels)
                val_acc.append(accuracy)
                print('val:',i, accuracy)
        print('validation average acc:',sum(val_acc)/len(val_acc))


def test():
    model.eval()
    test_acc = []
    for i, data in enumerate(test_loader):
        input_ids, attention_mask, token_type_ids, labels = collate_fn(data['text'], data['label'])
        input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(
            device), token_type_ids.to(device), labels.to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if i % 5 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            test_acc.append(accuracy)
            print('test:', i,  accuracy)
    print('test average acc:', sum(test_acc) / len(test_acc))

train(1)
test()
