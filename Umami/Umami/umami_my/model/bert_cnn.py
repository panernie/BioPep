from gensim.models import word2vec
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertPreTrainedModel, BertConfig, RobertaConfig, RobertaModel, AlbertConfig
from Umami.classification.umami_my.model.bert_encoder import CharBertEncoder
import torch.nn.functional as F
from torchsummary import summary
from torchviz import make_dot


class BertCNNModel(nn.Module):
    def  __init__(self):
        super(BertCNNModel, self).__init__()
        self.num_labels = 1
        self.hidden_size = 768
        self.sen_char_encoder = CharBertEncoder()
        model_config = BertConfig.from_pretrained("../process/iumami_model")
        self.window_sizes = [1,2,3]
        self.max_text_len = 29  #self.num_labels + 2
        self.bert = BertModel(model_config)
        self.dropout = nn.Dropout(0.1)
        self.dropout_rate = 0.1
        self.filter_size = 250
        self.dense_1 = nn.Linear(self.hidden_size, 1)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.hidden_size,
                                    out_channels=self.filter_size,
                                    kernel_size=h),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=self.max_text_len - h + 1))
            for h in self.window_sizes
        ])
        self.fc = nn.Linear(in_features=self.filter_size * len(self.window_sizes),
                            out_features=self.num_labels)

    def forward(self, inputs, token_type_ids=None, attention_mask=None, position_ids=None):
        outputs = self.bert(inputs, attention_mask=attention_mask)
        embed_x = outputs[0]
        embed_x = self.dropout(embed_x)

        embed_x = embed_x.permute(0, 2, 1)
        out = [conv(embed_x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))
        out = F.dropout(input=out, p=self.dropout_rate)
        out = self.fc(out).squeeze(1)
        out = out.sigmoid()

        return out



