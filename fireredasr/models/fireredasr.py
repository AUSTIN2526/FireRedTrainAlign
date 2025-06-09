import os
import time

import torch

from fireredasr.models.fireredasr_aed import FireRedAsrAed


def load_fireredasr_aed_model(model_path, weights_path):
    """
    Load an AED-based FireRed ASR model from a .pth.tar checkpoint.
    """
    
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)

    if weights_path is None:
        model_state_dict = package["model_state_dict"]
    else:
        model_state_dict = torch.load(weights_path)
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(model_state_dict, strict=True)
    return model


class FireRedAsr:
    """
    Simplified FireRed ASR class that only handles AED inference.
    Does not manage a tokenizer or feature extractor internally.
    """

    def __init__(self, model_name, weights_path = None):
        """
        Args:
            model_name (str): Path or identifier for the pretrained AED model.
                              Expects that under `model_name/` there is a
                              `model.pth.tar` checkpoint.
        """
        model_path = os.path.join(model_name, "model.pth.tar")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"AED checkpoint not found at {model_path}")
        self.model = load_fireredasr_aed_model(model_path, weights_path)
        self.model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        

    @torch.no_grad()
    def transcribe(self,
                   feats: torch.Tensor,
                   lengths: torch.Tensor,
                   durs: list,
                   beam_size: int = 1,
                   nbest: int = 1,
                   decode_max_len: int = 0,
                   softmax_smoothing: float = 1.0,
                   length_penalty: float = 0.0,
                   eos_penalty: float = 1.0):
        """
        Run AED-based transcription on precomputed features.

        Args:
            feats (Tensor): Padded feature batch, shape (B, T, D).
            lengths (Tensor): 1-D tensor of valid lengths for each item in the batch.
            durs (list of float): List of durations (in seconds) for each utterance.
            beam_size (int, optional): Beam size for decoding. Default: 1
            nbest (int, optional): Number of best hypotheses to return. Default: 1
            decode_max_len (int, optional): Maximum length for decoding. Default: 0 (no limit)
            softmax_smoothing (float, optional): Softmax smoothing factor. Default: 1.0
            length_penalty (float, optional): Length penalty for beam search. Default: 0.0
            eos_penalty (float, optional): EOS penalty for beam search. Default: 1.0

        Returns:
            List[Dict]: A list of hypothesis dictionaries (one per utterance). Each dict has keys:
                - "yseq": Tensor of token IDs for the best hypothesis (nbest=1 means one sequence).
                - (Additional information from the AED model can also appear in each dict.)
            float: Real-time factor (RTF) computed as (elapsed_time / total_duration).
        """
        # Move model and data to GPU if it’s already on GPU; otherwise, ensure CPU.
        device = next(self.model.parameters()).device
        feats = feats.to(device)
        lengths = lengths.to(device)

        start_time = time.time()
        hyps = self.model.transcribe(
            feats,
            lengths,
            beam_size,
            nbest,
            decode_max_len,
            softmax_smoothing,
            length_penalty,
            eos_penalty,
        )
        elapsed = time.time() - start_time
        total_dur = sum(durs) if isinstance(durs, (list, tuple)) else float(durs)
        rtf = elapsed / total_dur if total_dur > 0 else 0.0

        return hyps, rtf

    def forward(self,
                feats: torch.Tensor,
                lengths: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        对外暴露一个 forward()，用于训练时的 teacher‐forcing：
          - feats:   (B, T_enc, D)      —— 声学特征输入（已 pad 好），
                     例如 Mel 频谱帧序列。
          - lengths: (B,)                —— 每条序列对应的真实帧数。
          - targets: (B, T_tgt)          —— decoder 的输入序列，
                     通常已经在最前面放了 <sos>（即 sos_id），
                     末尾含 <eos> 并且多余部分用 <pad>（pad_id）补齐。

        返回：
          - logits:  (B, T_tgt, V)       —— 在每个时刻对下一个 token 的预测分数，
                     可以直接接交叉熵 loss 进行训练。
        """
        device = next(self.model.parameters()).device
        feats = feats.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        # 1. 先经 Encoder
        #    enc_outputs: (B, T_enc_sub, d_model)
        #    enc_lens:    (B,)  —— subsampling 之后的帧数
        #    enc_mask:    (B, 1, T_enc_sub)
        enc_outputs, enc_lens, enc_mask = self.model.encoder(feats, lengths)

        # 2. 再给 Decoder 计算 logits（teacher‐forcing 模式）
        #    targets 里应该已经带了 sos、eos 和 pad。
        logits = self.model.decoder(
            targets,       # (B, T_tgt)
            enc_outputs,   # (B, T_enc_sub, d_model)
            enc_mask       # (B, 1, T_enc_sub)
        )
        return logits
