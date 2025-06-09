import torch

from fireredasr.models.module.conformer_encoder import ConformerEncoder
from fireredasr.models.module.transformer_decoder import TransformerDecoder


class FireRedAsrAed(torch.nn.Module):
    @classmethod
    def from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super().__init__()
        self.sos_id = args.sos_id
        self.eos_id = args.eos_id

        self.encoder = ConformerEncoder(
            args.idim, args.n_layers_enc, args.n_head, args.d_model,
            args.residual_dropout, args.dropout_rate,
            args.kernel_size, args.pe_maxlen)

        self.decoder = TransformerDecoder(
            args.sos_id, args.eos_id, args.pad_id, args.odim,
            args.n_layers_dec, args.n_head, args.d_model,
            args.residual_dropout, args.pe_maxlen)

    def transcribe(self, padded_input, input_lengths,
                   beam_size=1, nbest=1, decode_max_len=0,
                   softmax_smoothing=1.0, length_penalty=0.0, eos_penalty=1.0):
        enc_outputs, _, enc_mask = self.encoder(padded_input, input_lengths)
        nbest_hyps = self.decoder.batch_beam_search(
            enc_outputs, enc_mask,
            beam_size, nbest, decode_max_len,
            softmax_smoothing, length_penalty, eos_penalty)
        return nbest_hyps
    
    def forward(self,
                padded_input,
                input_lengths,
                padded_target
            ):
        """
        训练时的前向计算：
          - padded_input:  (batch, T_waveform, feat_dim) 或者 (batch, T_frames, feat_dim)
          - input_lengths: (batch,) 表示每个序列的真实长度（帧数或采样点数）
          - padded_target: (batch, T_tgt) —— decoder 的输入序列（已加 sos，末尾有 eos + pad）
        返回：
          - logits:      (batch, T_tgt, odim)  —— decoder 在每个时刻对下一个 token 的预测分数
        """
        # 1. Encoder 计算
        #    enc_outputs: (batch, T_enc, d_model)
        #    enc_lengths: (batch,)     —— 经 subsampling 之后的真实长度
        #    enc_mask:    (batch, 1, T_enc)，0 表示 padding
        enc_outputs, enc_lengths, enc_mask = self.encoder(padded_input, input_lengths)

        # 2. Decoder 前向（teacher forcing）
        logits = self.decoder(
            padded_target,    # (batch, T_tgt)
            enc_outputs,      # (batch, T_enc, d_model)
            enc_mask          # (batch, 1, T_enc)
        )
        return logits