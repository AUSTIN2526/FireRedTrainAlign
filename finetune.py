import os
import random
import argparse
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, SequentialSampler
import kaldiio
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import math

from fireredasr.data.asr_feat import HFASRFeatExtractor
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from fireredasr.models.fireredasr import FireRedAsr


def parse_args():
    parser = argparse.ArgumentParser(description="Train FireRed ASR with dynamic weighted sampling")
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--label_file', type=str, default='task1_answer.txt')
    parser.add_argument('--model_name', type=str, default='FireRedASR-AED-L')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--use_validation', action='store_true')
    parser.add_argument('--use_weighted_sampling', action='store_true')
    parser.add_argument('--weight_update_interval', type=int, default=1)
    parser.add_argument('--difficulty_strategy', type=str, choices=['loss_based', 'length_based', 'unk_based', 'combined'], default='loss_based')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='firered_asr_output')
    return parser.parse_args()


def main():
    args = parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ========== 特徵提取與分詞器初始化 ==========
    feat_extractor = HFASRFeatExtractor(args.model_name)
    tokenizer = ChineseCharEnglishSpmTokenizer(args.model_name)
    pad_id, sos_id, eos_id, unk_id = 2, 3, 4, 1

    try:
        idim = feat_extractor.feat_dim
    except AttributeError:
        first_file = next(f for f in os.listdir(args.audio_dir) if f.lower().endswith(('.wav', '.flac')))
        sr_ex, wav_ex = kaldiio.load_mat(os.path.join(args.audio_dir, first_file))
        feat_ex, _, _ = feat_extractor((wav_ex, sr_ex))
        idim = feat_ex.shape[-1]

    wrapper = FireRedAsr(args.model_name)
    wrapper.model.to(DEVICE)

    # ========== 標註與 UNK 檢查 ==========
    results = {}
    with open(args.label_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            columns = line.strip().split('\t')
            if len(columns) != 2:
                raise ValueError(f"格式錯誤：{line!r}")
            key, value = columns
            results[key] = value

    sample_stats = {}
    for key, text in results.items():
        tokens, tokens_id = tokenizer.tokenize(text)
        sample_stats[key] = {
            'unk_count': sum(1 for tid in tokens_id if tid == unk_id),
            'text_length': len(tokens_id),
            'tokens': tokens,
            'tokens_id': tokens_id
        }

    # ========== Dataset 定義 ==========
    class ASRDataset(Dataset):
        def __init__(self, file_list):
            self.file_list = file_list

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, idx):
            fname = self.file_list[idx]
            base_key = os.path.splitext(fname)[0]
            filepath = os.path.join(args.audio_dir, fname)

            sr, waveform = kaldiio.load_mat(filepath)
            padded_feat, lengths, _ = feat_extractor((waveform, sr))
            tokens, tokens_id = tokenizer.tokenize(results[base_key])
            tgt_sequence = [sos_id] + tokens_id + [eos_id]
            return {
                "feat": padded_feat,
                "lengths": lengths,
                "decoder_input": torch.LongTensor(tgt_sequence[:-1]).unsqueeze(0),
                "gold_target": torch.LongTensor(tgt_sequence[1:]).unsqueeze(0),
                "file_key": base_key
            }

    def calculate_sample_weights(file_list, sample_stats, sample_losses=None, strategy="loss_based"):
        weights = []
        for fname in file_list:
            base_key = os.path.splitext(fname)[0]
            stats = sample_stats[base_key]
            if strategy == "loss_based" and sample_losses is not None:
                weight = sample_losses.get(base_key, 1.0)
            elif strategy == "length_based":
                weight = math.log(stats['text_length'] + 1)
            elif strategy == "unk_based":
                weight = stats['unk_count'] + 1
            elif strategy == "combined":
                weight = sample_losses.get(base_key, 1.0) * math.log(stats['text_length'] + 1) * (stats['unk_count'] + 1)
            else:
                weight = 1.0
            weights.append(weight)
        weights = np.array(weights)
        weights = weights / np.sum(weights) * len(weights)
        return weights.tolist()

    all_files = [f for f in os.listdir(args.audio_dir) if f.lower().endswith(('.wav', '.flac')) and os.path.splitext(f)[0] in results]
    random.shuffle(all_files)
    num_train = int(len(all_files) * args.train_ratio)
    train_files = all_files[:num_train]
    val_files = all_files[num_train:] if args.use_validation else []

    train_dataset = ASRDataset(train_files)
    val_dataset = ASRDataset(val_files) if args.use_validation else None

    if args.use_weighted_sampling:
        weights = calculate_sample_weights(train_files, sample_stats, strategy=args.difficulty_strategy)
        sampler = WeightedRandomSampler(weights, len(train_files), replacement=True)
    else:
        from torch.utils.data import RandomSampler
        sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=lambda b: b[0])
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=lambda b: b[0]) if val_dataset else None

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(wrapper.model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    sample_losses = defaultdict(list)

    for epoch in range(args.epochs):
        if args.use_weighted_sampling and epoch > 0 and epoch % args.weight_update_interval == 0:
            avg_sample_losses = {k: np.mean(v[-args.weight_update_interval:]) for k, v in sample_losses.items() if v}
            weights = calculate_sample_weights(train_files, sample_stats, avg_sample_losses, strategy=args.difficulty_strategy)
            sampler = WeightedRandomSampler(weights, len(train_files), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=lambda b: b[0])

        wrapper.model.train()
        total_train_loss = 0.0
        epoch_sample_losses = defaultdict(list)
        for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1} | Train]"):
            feat = batch['feat'].to(DEVICE)
            lengths = batch['lengths'].to(DEVICE)
            decoder_input = batch['decoder_input'].to(DEVICE)
            gold_target = batch['gold_target'].to(DEVICE)
            file_key = batch['file_key']

            logits = wrapper.forward(feat, lengths, decoder_input)
            logits = logits[:, :gold_target.size(1), :]

            loss = criterion(logits.view(-1, logits.size(-1)), gold_target.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_train_loss += loss_val
            epoch_sample_losses[file_key].append(loss_val)

        for k, v in epoch_sample_losses.items():
            sample_losses[k].extend(v)

        print(f"Epoch {epoch+1} Avg Train Loss: {total_train_loss/len(train_loader):.4f}")

        if val_loader:
            wrapper.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"[Epoch {epoch+1} | Val]"):
                    feat = batch['feat'].to(DEVICE)
                    lengths = batch['lengths'].to(DEVICE)
                    decoder_input = batch['decoder_input'].to(DEVICE)
                    gold_target = batch['gold_target'].to(DEVICE)

                    logits = wrapper.forward(feat, lengths, decoder_input)
                    logits = logits[:, :gold_target.size(1), :]

                    loss = criterion(logits.view(-1, logits.size(-1)), gold_target.view(-1))
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(wrapper.model.state_dict(), os.path.join(args.save_path, 'best_model.pth'))
                print("Model saved!")

    print("Training complete.")


if __name__ == '__main__':
    main()