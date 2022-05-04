from turtle import forward
from processFile import read_file, InputDataSet
from transformers import BertTokenizer, BertModel, BertPreTrainedModel,BertConfig, BertForPreTraining
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import logging
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import logit, nn
from d2l import torch as d2l
import torch
import os

logging.set_verbosity_error()


class TextCNN(nn.Module):
    def __init__(self, filter_sizes, num_filter, num_labels):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.num_filter_total = num_filter * len(filter_sizes)
        self.Weight = nn.Linear(self.num_filter_total, num_labels, bias=False)
        self.bias = nn.Parameter(torch.ones([num_labels]))
        self.filter_list = nn.ModuleList([
            nn.Conv2d(1, num_filter, kernel_size=(size, 768)) for size in filter_sizes
        ])
    
    def forward(self, x):
        x = x.unsqueeze(1)

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(x))
            mp = nn.MaxPool2d(kernel_size = (12 - self.filter_sizes[i] + 1, 1))
            pooled = mp(h).permute(0, 3, 2, 1) # [bs, h=1, w=1, channel=3]
            pooled_outputs.append(pooled)
        
        h_pool = torch.cat(pooled_outputs, 3) # [bs, h=1, w=1, channel=192]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])

        output = self.Weight(h_pool_flat) + self.bias # [bs, n_class]

        return output


class BertForSeq(BertPreTrainedModel):
    """Bert分类模型"""

    def __init__(self, config):
        super(BertForSeq, self).__init__(config)
        self.out_channels = 16
        # 获得预训练模型的参数
        self.config = BertConfig(config)
        # 标签数量
        self.num_labels = 31
        # bert模型
        self.bert = BertModel(config)   
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 最后的分类层
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # textCnn层
        self.text_cnn = TextCNN(filter_sizes=[2, 3, 4, 5], num_filter=64, num_labels=self.num_labels)

        # 初始化权重
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, return_dict=None):
        """前向传播"""
        # 若return_dict不是None的话则会返回一个字典，否则返回一个字符串
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取模型结果
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
            output_hidden_states=True
            )
        
        # bert的返回值有：last_hidden_state、pooler_output、hidden_states、attentions、cross_attentions、past_key_values
        # 取出pooler_output
        # pooled_output = outputs[1]     # [batch_size, hidden_size]
        # pooled_output = self.dropout(pooled_output)
        # pooled_output = pooled_output.unsqueeze(1)
        # pooled_output = self.text_cnn(pooled_output).view(batch_size, -1)   # [batch_size, -1]

        hidden_states = outputs.hidden_states # shape = (batch_size=16, sequence_length=512, hidden_size=768)
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)
        for i in range(2, 13):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # logits会返回一个还未经过
        logits = self.text_cnn(cls_embeddings)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("./金融评论分类/corpus/bert-base-uncased")
    model = BertForSeq.from_pretrained("./金融评论分类/corpus/bert-base-uncased")

    train = read_file("./金融评论分类/dataSet/train.csv")
    train_data = InputDataSet(train, tokenizer, 512)
    train_iter = DataLoader(train_data, batch_size=64, shuffle=False)
    batch = next(iter(train_iter))
    
    device = torch.device('cpu')

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    labels = batch['labels'].to(device)

    model.eval()

    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

    logits = nn.functional.softmax(outputs.logits, dim=1)
    loss = outputs.loss

    print(logits)
    print(loss.item())

    preds = torch.argmax(logits, dim=1)
    print(preds)