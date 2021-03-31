from math import sqrt
import numpy as np
from numpy import finfo
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths
from models.modules import GST
import sys


drop_rate = 0.5

def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    return model


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, attention_weights=None):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        if attention_weights is None:
            alignment = self.get_alignment_energies(
                attention_hidden_state, processed_memory, attention_weights_cat)

            if mask is not None:
                alignment.data.masked_fill_(mask, self.score_mask_value)

            attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights

class SMAttention(nn.Module):
    ''' 
    fix Attention to StepWise monotonic Attention.
    ref: from: https://github.com/keonlee9420/Stepwise_Monotonic_Multihead_Attention

    '''
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(SMAttention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")
        noise_std = 1
        self.noise_std = noise_std

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)

        energies = self.v(F.tanh(
            processed_query + processed_attention_weights + processed_memory))
        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask,attention_weights=None):
        """
        attention_rnn_dim, embedding_dim, attention_dim,
        attention_location_n_filters, attention_location_kernel_size
                
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs: B,T,memDim
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        # 
        attention_hidden_state:: - **h_1** of shape `(batch, hidden_size)
        self.attention_context, self.attention_weights = 
            self.attention_layer(
        self.attention_hidden, self.memory, self.processed_memory,attention_weights_cat, self.mask, attention_weights)
        # 
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),self.attention_weights_cum.unsqueeze(1)), dim=1)
        """
        # get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        # PARAMS
        # ------
        # query: decoder output (batch, n_mel_channels * n_frames_per_step)
        # processed_memory: processed encoder outputs (B, T_in, attention_dim)
        # attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        # RETURNS
        # -------
        # alignment (batch, max_time)
        mode = 'soft'
        if attention_weights is None:
            # 备注： alignment 就是 energy 
            alignment = self.get_alignment_energies(
                attention_hidden_state, processed_memory, attention_weights_cat)
            if mask is not None:
                alignment.data.masked_fill_(mask, self.score_mask_value)
            if self.noise_std > 0:
                alignment = alignment + alignment*self.add_gaussian_noise(alignment, self.noise_std) 
            # p_choose_i : attention_weights : p_sample
            if mode=='soft':
                # soft attention .  
                # attention_weights = F.softmax(alignment, dim=1)
                attention_weights = torch.sigmoid(alignment)
                # print('\n\t-->attention_weights.shape０=',attention_weights.shape)
                # print('\n\t-->attention_weights=０',attention_weights)
            else: # hard attention 
                y = torch.ones(alignment.shape,dtype=alignment.dtype)
                attention_weights = torch.where(alignment > 0,alignment,y)
        # 备注：　pre_att　初始化的时候，必须保证　第一帧对应到　第一个音素。　即：在　initialize_decoder_states　中　全零初始化，首列置１。
        # attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1),self.attention_weights_cum.unsqueeze(1)), dim=1)
        pre_att = attention_weights_cat[:,0,:]
        # 备注： batchSize= 32,  139　应该是　id 的长度。
        # attention_dim=128,
        # -->attention_weights.shape= torch.Size([32, 139])
        # -->attention_weights_cat.shape= torch.Size([32, 2, 139])
        # -->alignment.shape= torch.Size([32, 139])
        # -->processed_memory.shape= torch.Size([32, 139, 128]) 

        # -->attention_weights.shape= torch.Size([32, 133])
        # -->attention_weights_cat.shape= torch.Size([32, 2, 133])
        # -->processed_memory.shape= torch.Size([32, 133, 128]) 
        # mask_true = torch.where(torch.isnan(y_true), torch.full_like(y_true, 0), torch.full_like(y_true, 1))
        attention_weights = monotonic_stepwise_attention(attention_weights, pre_att, mode)
        # print('\n\t-->attention_weights.shape=',attention_weights.shape)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        # -->attention_weights.shape= torch.Size([32, 139])
        # -->attention_context.shape= torch.Size([32, 896])
        # -->memory.shape= torch.Size([32, 139, 896])

        return attention_context, attention_weights

    def add_gaussian_noise(self, xs, std):
        """Add Gaussian noise to encourage discreteness."""
        noise = xs.new_zeros(xs.size()).normal_(std=std)
        return xs + noise

def monotonic_stepwise_attention(p_choose_i, previous_attention, mode):
    # p_choose_i, previous_alignments, previous_score: [batch_size, memory_size]
    # p_choose_i: probability to keep attended to the last attended entry i
    # p_choose_i : B, T

    if mode == "soft":
        # pad = tf.zeros([tf.shape(p_choose_i)[0], 1], dtype=p_choose_i.dtype)
        pad = torch.zeros((p_choose_i.shape[0], 1), dtype=p_choose_i.dtype).to(p_choose_i.device)
        # attention = previous_attention * p_choose_i + tf.concat([pad, previous_attention[:, :-1] * (1.0 - p_choose_i[:, :-1])], axis=1)
        attention = previous_attention * p_choose_i + torch.cat((pad, previous_attention[:, :-1] * (torch.ones_like(p_choose_i[:, :-1]) - p_choose_i[:, :-1])), dim=1)
        # attention = previous_attention * p_choose_i
    
    elif mode == "hard":
        # Given that previous_alignments is one_hot
        # move_next_mask = tf.concat([tf.zeros_like(previous_attention[:, :1]), previous_attention[:, :-1]], axis=1)
        move_next_mask = torch.cat((torch.zeros_like(previous_attention[:, :1]), previous_attention[:, :-1]), dim=1)
        # stay_prob = tf.reduce_sum(p_choose_i * previous_attention, axis=1)  # [B]
        stay_prob = torch.sum(p_choose_i * previous_attention, dim=1)  # [B]
        # attention = tf.where(stay_prob > 0.5, previous_attention, move_next_mask)
        attention = torch.where(stay_prob > 0.5, previous_attention, move_next_mask)
    else:
        raise ValueError("mode must be 'parallel', or 'hard'.")
    return attention

    # SMA
    if 0:
        batch_size = attention_weights.shape[0]
        klen = attention_weights.shape[1] # encoder 的数目。
        # Compute probability sampling matrix P
        alpha = []
        # Compute recurrence relation solution along mel frame domain
        for i in range(klen):
            p_sample_i = p_sample[:, :, i:i + 1]
            pad = torch.zeros([batch_size, 1, 1], dtype=aw_prev.dtype).to(aw_prev.device)
            aw_prev = aw_prev * p_sample_i + torch.cat((pad, aw_prev[:, :-1, :] * (1.0 - p_sample_i[:, :-1, :])), dim=1)
            alpha.append(aw_prev)
        attention_weights = torch.cat(alpha, dim=-1) if klen > 1 else alpha[-1] # [batch*n_head, qlen, klen]
        assert not torch.isnan(attention_weights).any(), "NaN detected in alpha."

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=drop_rate, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        # x: mel_outputs:(B, n_mel_channels, T_out)
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), drop_rate, self.training)
        x = F.dropout(self.convolutions[-1](x), drop_rate, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        # x： embedded_inputs:  B,embDim,T
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), drop_rate, self.training)
        #x:  B,encoder_embedding_dim,T  --> B,T,encoder_embedding_dim
        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        # : outputs: B,T,lstm_dim
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), drop_rate, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim + hparams.token_embedding_size + hparams.speaker_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.p_teacher_forcing = hparams.p_teacher_forcing

        self.prenet_f0 = ConvNorm(
            1, hparams.prenet_f0_dim,
            kernel_size=hparams.prenet_f0_kernel_size,
            padding=max(0, int(hparams.prenet_f0_kernel_size/2)),
            bias=False, stride=1, dilation=1)

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.prenet_f0_dim + self.encoder_embedding_dim,
            hparams.attention_rnn_dim)
        # SMAttention(self, attention_rnn_dim, embedding_dim, attention_dim,
                #  attention_location_n_filters, attention_location_kernel_size):
        if  hparams.useSMA: 
            self.attention_layer = SMAttention(
                hparams.attention_rnn_dim, self.encoder_embedding_dim,
                hparams.attention_dim, hparams.attention_location_n_filters,
                hparams.attention_location_kernel_size)
        else:
            self.attention_layer = Attention(
                hparams.attention_rnn_dim, self.encoder_embedding_dim,
                hparams.attention_dim, hparams.attention_location_n_filters,
                hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + self.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def get_end_f0(self, f0s):
        B = f0s.size(0)
        dummy = Variable(f0s.data.new(B, 1, f0s.size(1)).zero_())
        return dummy

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs : B,T,
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        # 默认首帧对齐第一个音素。
        self.attention_weights[:,0] = 1

        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        # ,attention_dim
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs :(B, n_mel_channels, T_out)

        RETURNS
        -------
        inputs: processed decoder inputs : (T_out, B, n_mel_channels)

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:(T_out, B, n_mel_channels)
        gate_outputs: gate output energies    (T_out, B)
        alignments:(T_out, B)

        RETURNS
        -------
        mel_outputs:(B, n_mel_channels, T_out)
        gate_outpust: gate output energies   (B, T_out)
        alignments:(B, T_out)
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs)
        if len(gate_outputs.size()) > 1:
            gate_outputs = gate_outputs.transpose(0, 1)
        else:
            gate_outputs = gate_outputs[None]
        # transpose、permute 后使用 contiguous 方法则会重新
        # 开辟一块内存空间保证数据是在逻辑顺序和内存中是一致的，
        # 连续内存布局减少了CPU对对内存的请求次数（访问内存比访问寄存器慢100倍[5]），相当于空间换时间
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input, attention_weights=None):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        #
            # decoder_input ：  B行， mel_len + f0_len 列。
            # mel_outputs:(T_out, B, n_mel_channels)
            # gate_outputs: gate output energies    (T_out, B)
            # alignments:(T_out, B)
        """
        #　B行　特征＋atten-Vec 拼接。 
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        # cell_input: 维度＝hparams.prenet_dim + hparams.prenet_f0_dim + self.encoder_embedding_dim,
        # **input** of shape `(batch, input_size)
        # hparams.attention_rnn_dim)

        # attention_hidden:: - **h_1** of shape `(batch, hidden_size)`: tensor containing the next hidden state
        #   for each element in the batch
        # attention_cell:: - **c_1** of shape `(batch, hidden_size)`: tensor containing the next cell state
        #   for each element in the batch
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)
        self.attention_cell = F.dropout(
            self.attention_cell, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),self.attention_weights_cum.unsqueeze(1)), dim=1)

        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,attention_weights_cat, self.mask, attention_weights)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(
            self.decoder_cell, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)

        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths, f0s):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        # !!!
        mel_outputs, gate_outputs, alignments = self.decoder(
                    encoder_outputs, targets, memory_lengths=input_lengths, f0s=f0s)
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        # decoder_inputs: (B, n_mel_channels, T_out)-->  (T_out, B, n_mel_channels)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        # audio features
        f0_dummy = self.get_end_f0(f0s)
        f0s = torch.cat((f0s, f0_dummy), dim=2)
        f0s = F.relu(self.prenet_f0(f0s))
        f0s = f0s.permute(2, 0, 1)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        # 这种循环控制方式，保证输出的 mel帧数和 target_mel帧数一致。
        # decoder_inputs: (T_out, B, n_mel_channels)
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            # decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
            # 输入： B行， mel_len + f0_len 列。
            if len(mel_outputs) == 0 or np.random.uniform(0.0, 1.0) <= self.p_teacher_forcing:
                decoder_input = torch.cat((decoder_inputs[len(mel_outputs)],f0s[len(mel_outputs)]), dim=1)
            else:# 上一帧的解码输出。
                decoder_input = torch.cat((self.prenet(mel_outputs[-1]),f0s[len(mel_outputs)]), dim=1)
            # decoder_input ：  B行， mel_len + f0_len 列。
            # mel_outputs:(T_out, B, n_mel_channels)
            # gate_outputs: gate output energies    (T_out, B)
            # alignments:(T_out, B)
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]


        # mel_outputs:(B, n_mel_channels, T_out)
        # gate_outpust: gate output energies   (B, T_out)
        # alignments:(B, T_out)
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory, f0s):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)
        f0_dummy = self.get_end_f0(f0s)
        f0s = torch.cat((f0s, f0_dummy), dim=2)
        f0s = F.relu(self.prenet_f0(f0s))
        f0s = f0s.permute(2, 0, 1)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            if len(mel_outputs) < len(f0s):
                f0 = f0s[len(mel_outputs)]
            else:
                f0 = f0s[-1] * 0

            decoder_input = torch.cat((self.prenet(decoder_input), f0), dim=1)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference_noattention(self, memory, f0s, attention_map):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)
        f0_dummy = self.get_end_f0(f0s)

        # print('\t--> f0_dummy.shape=',f0_dummy.shape)
        # print('\t--> type(f0_dummy)=',type(f0_dummy))
        # print('\t--> f0_dummy=',f0_dummy,'\n')

        # print('\t--> f0s.shape=',f0s.shape)
        # print('\t--> type(f0s)=',type(f0s))
        # print('\t--> f0s=',f0s)


        f0s = torch.cat((f0s, f0_dummy), dim=2)

        f0s = F.relu(self.prenet_f0(f0s))
        f0s = f0s.permute(2, 0, 1)

        mel_outputs, gate_outputs, alignments = [], [], []
        for i in range(len(attention_map)):
            f0 = f0s[i]
            attention = attention_map[i]
            decoder_input = torch.cat((self.prenet(decoder_input), f0), dim=1)
            mel_output, gate_output, alignment = self.decode(decoder_input, attention)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        if hparams.with_gst:
            self.gst = GST(hparams)
        self.speaker_embedding = nn.Embedding(
            hparams.n_speakers, hparams.speaker_embedding_dim)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, speaker_ids, f0_padded = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        speaker_ids = to_gpu(speaker_ids.data).long()
        f0_padded = to_gpu(f0_padded).float()
        return ((text_padded, input_lengths, mel_padded, max_len,
                 output_lengths, speaker_ids, f0_padded),
                (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        '''  根据 output_lengths 对 outputs 进行 mask 然后输出。 '''
        # mel_outputs:(B, n_mel_channels, T_out)
        # gate_outpust_ energies  : (B, T_out)
        # alignments:(B, T_out)
        # ([mel_outputs, mel_outputs_postnet, gate_outputs, alignments],output_lengths)
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            #outputs[0]:  mel_outputs:(B, n_mel_channels, T_out)
            outputs[0].data.masked_fill_(mask, 0.0)
            # outputs[1] : mel_outputs_postnet : 
            outputs[1].data.masked_fill_(mask, 0.0)
            # outputs[2] : gate_outpust: (B, T_out)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        inputs, input_lengths, targets, max_len, \
            output_lengths, speaker_ids, f0s = inputs
        input_lengths, output_lengths = input_lengths.data, output_lengths.data
        # print('\t--models/model.py->inputs=',inputs)
        # print('\t--models/model.py->inputs.shape=',inputs.shape)
        # print('\t--models/model.py->input_lengths.shape=',input_lengths.shape)
        # print('\t--models/model.py->output_lengths.shape=',output_lengths.shape)

        # print('\t--models/model.py->targets.shape=',targets.shape)
        # print('\t--models/model.py->speaker_ids.shape=',speaker_ids.shape)
        # print('\t--models/model.py->speaker_ids=',speaker_ids)
        # print('\t--models/model.py->f0s.shape=',f0s.shape)

        # print('\t--models/model.py->type(inputs)=',type(inputs))
        # print('\t--models/model.py->inputs[0].shape=',inputs[0].shape)
        # print('\t--models/model.py->type(inputs[0])=',type(inputs[0]))

        # --models/model.py->inputs= tensor([[23, 18, 17]], device='cuda:0')
        # --models/model.py->inputs.shape= torch.Size([1, 3])
        # --models/model.py->type(inputs)= <class 'torch.Tensor'>
        # --models/model.py->inputs[0].shape= torch.Size([3])
        # --models/model.py->type(inputs[0])= <class 'torch.Tensor'>
        # --> f0.shape= torch.Size([1, 250])
        # --> type(f0)= <class 'torch.Tensor'>


        # inputs : B,T
        # embedded_inputs : B,T,embDim --> B,embDim,T
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        # embedded_text: B,T,lstm_dim
        embedded_text = self.encoder(embedded_inputs, input_lengths)
        embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
        embedded_gst = self.gst(targets, output_lengths)
        # toech.repeat 当参数只有两个时，第一个参数表示的是行复制的次数，第二个参数表示列复制的次数；
        # 当参数有三个时，第一个参数表示的是通道复制的次数，第二个参数表示的是行复制的次数，第三个参数表示列复制的次数
        embedded_gst = embedded_gst.repeat(1, embedded_text.size(1), 1)
        embedded_speakers = embedded_speakers.repeat(1, embedded_text.size(1), 1)

        # 此处说明：　text + spk_id + gst 拼接在一起　组成　decoder的　输入。
        encoder_outputs = torch.cat(
            (embedded_text, embedded_gst, embedded_speakers), dim=2)
        # mel_outputs:(B, n_mel_channels, T_out)
        # gate_outpust_ energies  : (B, T_out)
        # alignments:(B, T_out)
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths, f0s=f0s)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        # sys.exit()
        # 根据 output_lengths 对 outputs 进行 mask 然后输出。
        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs):
        text, style_input, speaker_ids, f0s = inputs
        embedded_inputs = self.embedding(text).transpose(1, 2)
        embedded_text = self.encoder.inference(embedded_inputs)
        embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
        if hasattr(self, 'gst'):
            if isinstance(style_input, int):
                query = torch.zeros(1, 1, self.gst.encoder.ref_enc_gru_size).cuda()
                GST = torch.tanh(self.gst.stl.embed)
                key = GST[style_input].unsqueeze(0).expand(1, -1, -1)
                embedded_gst = self.gst.stl.attention(query, key)
            else:
                embedded_gst = self.gst(style_input)

        embedded_speakers = embedded_speakers.repeat(1, embedded_text.size(1), 1)
        if hasattr(self, 'gst'):
            embedded_gst = embedded_gst.repeat(1, embedded_text.size(1), 1)
            encoder_outputs = torch.cat(
                (embedded_text, embedded_gst, embedded_speakers), dim=2)
        else:
            encoder_outputs = torch.cat(
                (embedded_text, embedded_speakers), dim=2)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, f0s)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

    def inference_noattention(self, inputs):
        text, style_input, speaker_ids, f0s, attention_map = inputs

        print('\t--models/model.py->text=',text)
        print('\t--models/model.py->text.shape=',text.shape)
        print('\t--models/model.py->type(text)=',type(text))

        embedded_inputs = self.embedding(text).transpose(1, 2)
        embedded_text = self.encoder.inference(embedded_inputs)
        embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
        if hasattr(self, 'gst'):
            if isinstance(style_input, int):
                query = torch.zeros(1, 1, self.gst.encoder.ref_enc_gru_size).cuda()
                GST = torch.tanh(self.gst.stl.embed)
                key = GST[style_input].unsqueeze(0).expand(1, -1, -1)
                embedded_gst = self.gst.stl.attention(query, key)
            else:
                embedded_gst = self.gst(style_input)

        embedded_speakers = embedded_speakers.repeat(1, embedded_text.size(1), 1)
        if hasattr(self, 'gst'):
            embedded_gst = embedded_gst.repeat(1, embedded_text.size(1), 1)
            encoder_outputs = torch.cat(
                (embedded_text, embedded_gst, embedded_speakers), dim=2)
        else:
            encoder_outputs = torch.cat(
                (embedded_text, embedded_speakers), dim=2)

        mel_outputs, gate_outputs, alignments = self.decoder.inference_noattention(
            encoder_outputs, f0s, attention_map)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
