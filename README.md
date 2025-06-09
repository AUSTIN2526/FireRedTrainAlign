# 🔥 FireRedTrainAlign：中文語音辨識訓練與推論套件

本專案提供針對 [FireRedTeam/FireRedASR-AED-L](https://huggingface.co/FireRedTeam/FireRedASR-AED-L) 模型的訓練與推論程式碼，支援高品質的中文語音轉錄任務。  
模型採用 **Attention-based Encoder-Decoder（AED）** 架構，適用於普通話及部分方言的語音辨識情境。

---

## 🚀 專案特色

- ✅ 支援 Hugging Face 上的 `FireRedASR-AED-L` 模型
- ✅ 提供完整訓練流程（資料前處理、模型訓練與應用）
- ✅ 支援 Beam Search 解碼、多音檔批次推論
- ✅ 模型對中長語音段落進行優化，適用於多種中文語音場景

---

## ⚙️ 安裝與環境需求

### 1️⃣ 建立虛擬環境（建議使用 Conda）

```bash
git clone https://github.com/AUSTIN2526/FireRedTrainAlign.git
cd FireRedTrainAlign

conda create -n firered_asr python=3.10
conda activate firered_asr

pip install -r requirements.txt
````

### 2️⃣ 額外依賴

* `ffmpeg`：用於音訊轉檔與取樣率轉換
* `CUDA`（選用）：若需 GPU 加速訓練或推論

---

## 🎯 注意事項與限制

* 建議語音長度 **不超過 60 秒**，以避免解碼錯誤或語句重複
* 音訊長度若 **超過 200 秒**，可能導致位置編碼（positional encoding）錯誤
* Batch 推論與訓練目前尚未全面測試
* 請將模型 `FireRedASR-AED-L` 放置於專案資料夾中使用

---

## 📁 訓練資料準備方式

請準備如下格式的 `.tsv` 文字檔，每一列表示一筆語音資料與對應轉錄文字：

```
ID\t需轉錄文字
ID\t需轉錄文字
ID\t需轉錄文字
...
```

---

## 📊 模型表現（來自官方論文）

| 模型名稱             | 參數數量 | AISHELL1 | AISHELL2 | WS Net | WS Meeting | 平均 CER    |
| ---------------- | ---- | -------- | -------- | ------ | ---------- | --------- |
| FireRedASR-AED-L | 1.1B | 0.55%    | 2.52%    | 4.88%  | 4.76%      | **3.18%** |

---
