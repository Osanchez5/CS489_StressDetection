import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from Modules.Embed import DataEmbedding
from Modules.Conv_Blocks import Inception_Block_V1


# Initial layout for the model, will definitely be changed as we do further research on Transformer models and TimesNet
# Referenced the original that initially proposed the TimesNet model. Code is also referenced from that same article
# Original article had different use cases for TimesNet, we probably only have to worry about classification more than anything
# They separated the TimesNet model and the TimesBlock into separate classes
class TimesNet(nn.Module):
    def __init__(self, configs):
        super(TimesNet, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embd, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        # classification will be the default, unless we find another task that suits the dataset more
        self.actFunc = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)

    # Probably won't be needed since it only serves to make selection between different tasks
    # Easier in the source code
    # def classification(self, x_enc, x_mark_enc):
    #     # Will be changed depending on our needs most likely
    #     enc_out = self.enc_embedding(x_enc, None)
    #     for i in range(self.layer):
    #         enc_out = self.layer_norm(self.model[i](enc_out))
    #     output = self.act(enc_out)
    #     output = self.dropout(output)

    #     # Zero-out padding embeddings
    #     output = output * x_mark_enc.unsqueeze(-1)

    #     output = output.reshape(output.shape[0], -1)
    #     output = self.projection(output)
    #     return output

    # def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # # Normalization from Non-stationary Transformer at temporal dimension
        # means = x_enc.mean(1, keepdim=True).detach() #[B,T]
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        # # embedding: projecting a number to a C-channel vector
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C] C is d_model
        # enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
        #     0, 2, 1)  # align temporal dimension [B,pred_len+seq_len,C]
        
        # # TimesNet: pass through TimesBlock for self.layer times each with layer normalization
        # for i in range(self.layer):
        #     enc_out = self.layer_norm(self.model[i](enc_out))

        # # project back  #[B,T,d_model]-->[B,T,c_out]
        # dec_out = self.projection(enc_out) 

        # # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out * \
        #         (stdev[:, 0, :].unsqueeze(1).repeat(
        #             1, self.pred_len + self.seq_len, 1)) #lengthen the stdev to fit the dec_out
        # dec_out = dec_out + \
        #         (means[:, 0, :].unsqueeze(1).repeat(
        #             1, self.pred_len + self.seq_len, 1)) #lengthen the mean to fit the dec_out
    #     return dec_out

    def forward(self, x_enc, x_mark_enc):
        # dec_out = self.classification(x_enc, x_mark_enc)

        # Update to add forecast code


        # enc_out = self.enc_embedding(x_enc, None)
        # for i in range(self.layer):
        #     enc_out = self.layer_norm(self.model[i](enc_out))
        # output = self.actFunc(enc_out)
        # output = self.dropout(output)

        # # Zero-out padding embeddings
        # output = output * x_mark_enc.unsqueeze(-1)

        # output = output.reshape(output.shape[0], -1)
        # x = self.projection(output)

        # This is the forecast version, simply comment out this portion and uncomment the portion above
        # classification is desired
        # Normalization from Non-stationary Transformer at temporal dimension
        means = x_enc.mean(1, keepdim=True).detach() #[B,T]
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding: projecting a number to a C-channel vector
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C] C is d_model
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension [B,pred_len+seq_len,C]
        
        # TimesNet: pass through TimesBlock for self.layer times each with layer normalization
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # project back  #[B,T,d_model]-->[B,T,c_out]
        dec_out = self.projection(enc_out) 

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                (stdev[:, 0, :].unsqueeze(1).repeat(
                    1, self.pred_len + self.seq_len, 1)) #lengthen the stdev to fit the dec_out
        x = dec_out + \
                (means[:, 0, :].unsqueeze(1).repeat(
                    1, self.pred_len + self.seq_len, 1)) #lengthen the mean to fit the dec_out

        return x
    
    

class TimesBlock(nn.Module):
    # Configs will have to be predefined or the code needs to be changed
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        # Will most likely be modified as we figure out the project
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        self.conv = nn.Sequential(
            # Inception block
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )
    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # Padding done here
            if(self.seq_len + self.pred_len) % period != 0:
                length = ( ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshaping
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()

            # 2D convolution with inception blocks
            out = self.conv(out)
            # Reshape to original 
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:,:(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # Adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)

        res = torch.sum(res * period_weight, -1) # have to combine in order to have same shape as x for combining in next line
        res = res + x
        return res

def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)

    # Find the period by the amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]