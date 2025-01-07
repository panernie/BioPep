import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import random
from tqdm import tqdm
import esm

seed_value = 42
random.seed(seed_value)                         # 设置 random 模块的随机种子
np.random.seed(seed_value)                      # 设置 numpy 模块的随机种子
torch.manual_seed(seed_value)                   # 设置 PyTorch 中 CPU 的随机种子
#tf.random.set_seed(seed_value)                 # 设置 Tensorflow 中随机种子
if torch.cuda.is_available():                   # 如果可以使用 CUDA，设置随机种子
    torch.cuda.manual_seed(seed_value)          # 设置 PyTorch 中 GPU 的随机种子
# 检测GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ESM-2 model
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D() #320d
batch_converter = alphabet.get_batch_converter()
esm_model.eval()  # disables dropout for deterministic results

def ESM_feature(sequence):
    data = []
    for i in tqdm(sequence):
        row = (i,i)
        data.append(row)

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[6], return_contacts=True)
    token_representations = results["representations"][6]
# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    return sequence_representations

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TransformerModel, self).__init__()
        # 构建Transformer编码层，参数包括输入维度、注意力头数
        # 其中d_model要和模型输入维度相同
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size,  # 输入维度
                                                        batch_first=True,
                                                        nhead=8)             # 注意力头数
        # 构建Transformer编码器，参数包括编码层和层数
        self.encoder = nn.TransformerEncoder(self.encoder_layer,             # 编码层
                                             num_layers=10)                   # 层数
        # 构建线性层，参数包括输入维度和输出维度（num_classes）
        self.fc = nn.Linear(input_size,                                      # 输入维度
                            num_classes)                                     # 输出维度


    def forward(self, x):
        #print("A:", x.shape)  # torch.Size([142, 13])
        x = x.unsqueeze(1)    # 增加一个维度，变成(batch_size, 1, input_size)的形状
        #print("B:", x.shape)  # torch.Size([142, 1, 13])
        x = self.encoder(x)   # 输入Transformer编码器进行编码
        #print("C:", x.shape)  # torch.Size([142, 1, 13])
        x = x.squeeze(1)      # 压缩第1维，变成(batch_size, input_size)的形状
        #print("D:", x.shape)  # torch.Size([142, 13])
        x = self.fc(x)        # 输入线性层进行分类预测
        #print("E:", x.shape)  # torch.Size([142, 3])
        return x
    
df=pd.read_excel("D:\Desk\BioPeptide_model\input_data\input.xlsx", na_filter=False)
len_num = [len(num) for num in df["Peptide"]]
df["len_num"] = len_num


RF_sequence = df[df["len_num"]>10]["Peptide"]
transformer_sequence = df[df["len_num"]<11]["Peptide"]

if len(RF_sequence)!=0:
    RF_sequence_representations = ESM_feature(RF_sequence)
    transformer_sequence_representations = ESM_feature(transformer_sequence)
    import joblib
    scaler_filename =r"model\Standard_scaler.save"
    RF_model = joblib.load(r"model\RandomForest_classf.pkl")
    model = torch.load(r'model\transformer_class.pt',map_location=torch.device('cuda:0'))#.to(device)

    scaler = joblib.load(scaler_filename)
    RF_pred = torch.stack(RF_sequence_representations)
    transformer_X_pred = torch.stack(transformer_sequence_representations)
    RF_X_pred = scaler.transform(RF_pred)
    transformer_X_pred = scaler.transform(transformer_X_pred)

    RF_y = RF_model.predict(RF_pred)
    RF_y_pro = RF_model.predict_proba(RF_pred)
    RF_df = pd.DataFrame(np.array(RF_sequence),columns=["Peptide"])
    RF_df["Active_pro"] = RF_y_pro[:,0]
    RF_df["Inactive_pro"] = RF_y_pro[:,1]

    with torch.no_grad():
        model.eval()
        y_hat = model(torch.tensor(transformer_X_pred).float().to(device))   # 使用训练好的模型对测试集进行预测
        y_score = torch.softmax(y_hat, dim=1).data.cpu().numpy()
        prediction = torch.max(F.softmax(y_hat,dim=1), 1)[1]
        #pred_y = prediction.data.cpu().numpy().squeeze()

    transformer_df = pd.DataFrame(np.array(transformer_sequence),columns=["Peptide"])
    transformer_df["Active_pro"] = y_score[:,0]
    transformer_df["Inactive_pro"] = y_score[:,1]
    df_all = pd.concat([RF_df,transformer_df], axis=0)
    df_all['ACE_Label'] = df_all['Active_pro'].apply(lambda x: 'Active' if x > 0.5 else 'Non-Active')
    df_all_ok = df_all[["Peptide","ACE_Label"]]
    df_all_ok.to_excel("D:\Desk\BioPeptide_model\output_result\ACE_peptide_result.xlsx",index=None)

else: 
    #RF_sequence_representations = ESM_feature(RF_sequence)
    transformer_sequence_representations = ESM_feature(transformer_sequence)
    import joblib
    scaler_filename =r"model\Standard_scaler.save"
    #RF_model = joblib.load(r"model\RandomForest_classf.pkl")
    model = torch.load(r'model\transformer_class.pt',map_location=torch.device('cuda:0'))#.to(device)

    scaler = joblib.load(scaler_filename)
    #RF_pred = torch.stack(RF_sequence_representations)
    transformer_X_pred = torch.stack(transformer_sequence_representations)
    #RF_X_pred = scaler.transform(RF_pred)
    transformer_X_pred = scaler.transform(transformer_X_pred)

    #RF_y = RF_model.predict(RF_pred)
    #RF_y_pro = RF_model.predict_proba(RF_pred)
    #RF_df = pd.DataFrame(np.array(RF_sequence),columns=["Peptide"])
    #RF_df["Active_pro"] = RF_y_pro[:,0]
    #RF_df["Inactive_pro"] = RF_y_pro[:,1]

    with torch.no_grad():
        model.eval()
        y_hat = model(torch.tensor(transformer_X_pred).float().to(device))   # 使用训练好的模型对测试集进行预测
        y_score = torch.softmax(y_hat, dim=1).data.cpu().numpy()
        prediction = torch.max(F.softmax(y_hat,dim=1), 1)[1]
        #pred_y = prediction.data.cpu().numpy().squeeze()

    transformer_df = pd.DataFrame(np.array(transformer_sequence),columns=["Peptide"])
    transformer_df["Active_pro"] = y_score[:,0]
    transformer_df["Inactive_pro"] = y_score[:,1]
    #df_all = pd.concat([RF_df,transformer_df], axis=0)
    transformer_df['ACE_Label'] = transformer_df['Active_pro'].apply(lambda x: 'Active' if x > 0.5 else 'Non-Active')
    df_all_ok = transformer_df[["Peptide","ACE_Label"]]
    df_all_ok.to_excel("D:\Desk\BioPeptide_model\output_result\ACE_peptide_result.xlsx",index=None)



