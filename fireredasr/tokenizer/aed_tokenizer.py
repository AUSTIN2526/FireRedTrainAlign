import logging
import re
import os

import sentencepiece as spm

from fireredasr.data.token_dict import TokenDict


class ChineseCharEnglishSpmTokenizer:
    """
    - One Chinese char is a token.
    - Split English word into SPM and one piece is a token.
    - Ignore ' ' between Chinese char
    - Replace ' ' between English word with "▁" by spm_model
    - Need to put SPM piece into dict file
    - If not set spm_model, will use English char and <space>
    
    初始化时传入一个目录（里面必须有 train_bpe1000.model 和 dict.txt），
    tokenize / detokenize 的逻辑保持原样不变。
    """
    SPM_SPACE = "▁"

    def __init__(self,
                 tokenizer_dir: str,
                 unk: str = "<unk>",
                 space: str = "<space>"):
        """
        Args:
            tokenizer_dir: 包含以下两个文件的目录
                - train_bpe1000.model
                - dict.txt
            unk: 未登录词在 dict.txt 中的表示（默认 "<unk>"）
            space: 如果没有 SPM，要用的空格符号（默认 "<space>"）
        """
        # --- 只修改这一部分：从目录里找 train_bpe1000.model 和 dict.txt ---
        model_path = os.path.join(tokenizer_dir, "train_bpe1000.model")
        dict_path = os.path.join(tokenizer_dir, "dict.txt")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"找不到 SentencePiece 模型：{model_path}")
        if not os.path.isfile(dict_path):
            raise FileNotFoundError(f"找不到词表文件：{dict_path}")
        # -------------------------------------------------------------------

        # 原本的逻辑：加载 TokenDict 和 SPM
        self.dict = TokenDict(dict_path, unk=unk)
        self.space = space
        if spm is not None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(model_path)
        else:
            self.sp = None
            print("[WARN] Not set spm_model, will use English char")
            print("[WARN] Please check how to deal with ' '(space)")
            if self.space not in self.dict:
                print("Please add <space> to your dict, or it will be <unk>")

        # 保持原有的正则，用于匹配“中文单字”
        self._zh_pattern = re.compile(r'([\u3400-\u4dbf\u4e00-\u9fff])')

    def tokenize(self, text, replace_punc=True):
        #if text == "":
        #    logging.info(f"empty text")
        text = text.upper()
        tokens = []
        if replace_punc:
            text = re.sub("[，。？！,\.?!]", " ", text)
        parts = self._zh_pattern.split(text.strip())
        parts = [p for p in parts if len(p.strip()) > 0]
        for part in parts:
            if self._zh_pattern.fullmatch(part) is not None:
                tokens.append(part)
            else:
                if self.sp:
                    for piece in self.sp.EncodeAsPieces(part.strip()):
                        tokens.append(piece)
                else:
                    for char in part.strip():
                        tokens.append(char if char != " " else self.space)
        tokens_id = []
        for token in tokens:
            tokens_id.append(self.dict.get(token, self.dict.unk))
        return tokens, tokens_id

    def detokenize(self, inputs, join_symbol="", replace_spm_space=True):
        """inputs is ids or tokens, do not need self.sp"""
        if len(inputs) > 0 and type(inputs[0]) == int:
            tokens = [self.dict[id] for id in inputs]
        else:
            tokens = inputs
        s = f"{join_symbol}".join(tokens)
        if replace_spm_space:
            s = s.replace(self.SPM_SPACE, ' ').strip()
        return s
