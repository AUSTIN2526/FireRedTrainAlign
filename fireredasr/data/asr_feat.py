import os
import math
from typing import Union, List, Tuple

import kaldiio
import kaldi_native_fbank as knf
import numpy as np
import torch


class HFASRFeatExtractor:
    """
    一个“类 Hugging Face”风格的 ASR 特征提取器。
    1. 初始化时传入一个目录（里面应该包含 cmvn.ark 或 cmvn.txt），
       会自动加载 CMVN 参数并初始化 Fbank 计算器。
    2. 调用时，直接传入 numpy 波形（或一组 numpy 波形）和采样率，
       会输出 pad 后的 fbank 特征张量、每条语音的帧长度列表，以及每条语音的时长（秒）。
    """
    def __init__(self, feature_dir: str, num_mel_bins: int = 80, frame_length: float = 25.0,
                 frame_shift: float = 10.0, dither: float = 0.0):
        """
        Args:
            feature_dir: 包含 CMVN 文件（如 global_cmvn.ark / global_cmvn.txt）的目录。
            num_mel_bins: Mel filter 数目（默认 80）
            frame_length: 每帧时长，单位 ms（默认 25ms）
            frame_shift: 帧移，单位 ms（默认 10ms）
            dither: 加噪参数，通常在推理阶段设为 0.0（默认 0.0）
        """
        # 先尝试从 feature_dir 里加载 CMVN stats
        cmvn_path = self._find_cmvn_file(feature_dir)
        if cmvn_path is not None:
            self.cmvn = _CMVN(cmvn_path)
        else:
            self.cmvn = None  # 如果目录里没有 cmvn 文件，就不做归一化

        # 初始化 Fbank 计算器
        self.fbank = _KaldifeatFbank(
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither
        )

    def _find_cmvn_file(self, folder: str) -> Union[str, None]:
        """
        在指定文件夹下查找 Kaldi CMVN 统计文件（.ark 或 .txt）。
        返回第一个能被 kaldiio.load_mat 读取的路径；如果都没有，则返回 None。
        """
        # 常见命名可能是 global_cmvn.ark 或 global_cmvn.txt，或者其他任意后缀为 .ark/.txt 且可以 load_mat
        for fname in os.listdir(folder):
            if fname.endswith(".ark") or fname.endswith(".txt"):
                candidate = os.path.join(folder, fname)
                try:
                    _ = kaldiio.load_mat(candidate)
                    return candidate
                except Exception:
                    continue
        return None

    def __call__(
        self,
        waves: Union[Tuple[np.ndarray, int], List[Tuple[np.ndarray, int]]]
    ) -> Tuple[torch.Tensor, torch.LongTensor, List[float]]:
        """
        传入单个或一组( wave_np, sample_rate )，输出特征 (padded_feats, lengths, durations)。

        Args:
            waves: 
                - 如果是单条语音：传入 (wav_np: np.ndarray, sample_rate: int)
                  其中 wav_np.shape=(T,), dtype=float32 或 float64，值范围 [-1.0, 1.0]（或其他归一/未归一格式都可以，只要一致即可）。
                - 如果是多条语音：传入 List[ (wav_np, sample_rate), ... ]。

        Returns:
            - padded_feats: torch.Tensor，shape = (batch_size, max_frames, num_mel_bins)
              已做 CMVN（如果提供）和 padding，类型 float32。
            - lengths: torch.LongTensor，shape=(batch_size,)，每条语音的帧数（不含 pad 部分）。
            - durations: List[float]，每条语音的时长，单位秒 = wave_np.shape[0] / sample_rate。
        """
        is_list_input = isinstance(waves, list)
        if not is_list_input:
            waves = [waves]

        feats_list = []
        lengths_list = []
        durations = []

        for (wav_np, sample_rate) in waves:
            # wave_np 必须是一维 np.ndarray
            assert isinstance(wav_np, np.ndarray) and wav_np.ndim == 1, \
                "传入的 wav 必须是 shape=(T,) 的一维 numpy 数组"

            # 计算时长（秒）
            dur = wav_np.shape[0] / float(sample_rate)
            durations.append(dur)

            # 计算原始 fbank 特征 (num_frames, num_mel_bins)
            fbank_feat = self.fbank((sample_rate, wav_np))
            # 如果有 cmvn，就做 CMVN 校正
            if self.cmvn is not None:
                fbank_feat = self.cmvn(fbank_feat)
            # 转成 torch.Tensor
            fbank_torch = torch.from_numpy(fbank_feat).float()
            feats_list.append(fbank_torch)
            lengths_list.append(fbank_torch.size(0))

        # lengths张量
        lengths = torch.tensor(lengths_list, dtype=torch.long)

        # pad 到 batch 里最大帧数
        padded = self._pad_feats(feats_list, pad_value=0.0)

        return padded, lengths, durations

    def _pad_feats(self, xs: List[torch.Tensor], pad_value: float) -> torch.Tensor:
        """
        内部方法，把一组 [Tensor_i: (T_i, num_mel_bins)] padding 到同样长度，
        返回 shape=(batch, max_T, num_mel_bins) 的 Tensor。
        """
        batch_size = len(xs)
        max_len = max(x.size(0) for x in xs)
        # 假设每个 feat.shape = (T_i, D)，D = num_mel_bins
        D = xs[0].size(1)
        device = xs[0].device
        dtype = xs[0].dtype

        padded = xs[0].new_full((batch_size, max_len, D), pad_value)
        for i, feat in enumerate(xs):
            padded[i, : feat.size(0), :] = feat
        return padded


class _CMVN:
    """
    Kaldi-style的CMVN (Cepstral Mean & Variance Normalization) 读取与应用类，
    接收一个 kaldi cmvn.ark 或 cmvn.txt（文本ARK）文件路径，内部用 kaldiio.load_mat 解析。
    """
    def __init__(self, kaldi_cmvn_path: str):
        self.dim, self.means, self.inv_std = self._load_cmvn(kaldi_cmvn_path)

    def _load_cmvn(self, path: str) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        读取 kaldi cmvn stats：
        - 文件格式是 2×(D+1) 的矩阵，第一行是 sum，第二行是 sumsq，最后一列是计数（count）。
        - D 是特征维度。
        返回 (D, means_array, inv_std_array)。
        """
        assert os.path.exists(path), f"CMVN 文件不存在: {path}"
        stats = kaldiio.load_mat(path)  # shape = (2, D+1)
        assert stats.ndim == 2 and stats.shape[0] == 2, "Kaldi CMVN stats 格式不正确"
        D = stats.shape[1] - 1
        count = stats[0, D]
        assert count >= 1.0, "CMVN 计数必须大于等于 1"

        means = np.zeros(D, dtype=np.float64)
        inv_std = np.zeros(D, dtype=np.float64)
        floor = 1e-20  # 防止除零

        for d in range(D):
            m = stats[0, d] / count
            means[d] = m
            var = stats[1, d] / count - m * m
            if var < floor:
                var = floor
            inv_std[d] = 1.0 / math.sqrt(var)
        return D, means.astype(np.float32), inv_std.astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        对单个语音的 fbank 特征做 CMVN： (T, D) -> (T, D)
        x.shape[-1] 必须等于 self.dim
        """
        assert x.ndim == 2 and x.shape[1] == self.dim, "CMVN 输入维度不匹配"
        # (T, D) - (D,) = (T, D)
        out = x - self.means[np.newaxis, :]
        # (T, D) * (D,) = (T, D)
        out = out * self.inv_std[np.newaxis, :]
        return out


class _KaldifeatFbank:
    """
    包装了 kaldi_native_fbank.OnlineFbank，用于从原始一维波形生成 fbank 特征。
    """
    def __init__(self, num_mel_bins: int = 80, frame_length: float = 25.0,
                 frame_shift: float = 10.0, dither: float = 1.0):
        """
        Args:
            num_mel_bins: mel filter 数目
            frame_length: 单位 ms
            frame_shift: 单位 ms
            dither: 加噪大小（训练时可设为 1.0，推理时设为 0.0）
        """
        opts = knf.FbankOptions()
        opts.frame_opts.dither = dither
        opts.mel_opts.num_bins = num_mel_bins
        opts.frame_opts.frame_length_ms = frame_length
        opts.frame_opts.frame_shift_ms = frame_shift
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts
        self.dither = dither

    def __call__(self, wav: Tuple[int, np.ndarray], is_train: bool = False) -> np.ndarray:
        """
        Args:
            wav: (sample_rate, wave_data)，wave_data 是一维 numpy 数组
            is_train: 是否在训练模式（此时使用 self.dither）；若为 False，则不加噪
        Returns:
            numpy.ndarray: shape = (num_frames, num_mel_bins)
        """
        sample_rate, wav_np = wav
        assert isinstance(wav_np, np.ndarray) and wav_np.ndim == 1, "输入 wave 必须是一维 numpy 数组"

        # 根据 is_train 决定 dither 大小
        dither = self.dither if is_train else 0.0
        self.opts.frame_opts.dither = dither

        fbank_inst = knf.OnlineFbank(self.opts)
        fbank_inst.accept_waveform(sample_rate, wav_np.tolist())

        frames = []
        for i in range(fbank_inst.num_frames_ready):
            frames.append(fbank_inst.get_frame(i))
        if len(frames) == 0:
            # 如果没有任何帧，则返回全零
            N = 0
            D = self.opts.mel_opts.num_bins
            return np.zeros((N, D), dtype=np.float32)
        feat = np.vstack(frames).astype(np.float32)
        return feat
