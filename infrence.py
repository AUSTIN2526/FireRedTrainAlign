import os
import json
import argparse
import torch
import kaldiio
import whisperx
from fireredasr.data.asr_feat import HFASRFeatExtractor
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from fireredasr.models.fireredasr import FireRedAsr


def transcribe_and_align(input_path, model, feat_extractor, tokenizer, align_model, metadata, device):
    sr, wav_np = kaldiio.load_mat(input_path)
    padded_feat, lengths, durs = feat_extractor((wav_np, sr))

    hyps_list, _ = model.transcribe(
        padded_feat,
        lengths,
        durs,
        beam_size=20,
        nbest=1,
        decode_max_len=0,
        softmax_smoothing=1.0,
        length_penalty=0.1,
        eos_penalty=1.0,
    )

    best_hyp = hyps_list[0][0]
    token_ids = [int(t) for t in best_hyp["yseq"].cpu().tolist()]
    transcript = tokenizer.detokenize(token_ids)

    print(f"[TEXT] {transcript}")

    initial_segments = [{
        "text": transcript,
        "start": 0.0,
        "end": durs[0]
    }]

    result_aligned = whisperx.align(
        initial_segments,
        align_model,
        metadata,
        input_path,
        device=device,
        return_char_alignments=True
    )

    output_segments = []
    global_idx = 0
    for seg in result_aligned["segments"]:
        for w in seg["words"]:
            output_segments.append({
                "id": global_idx,
                "start": float(w["start"]),
                "end": float(w["end"]),
                "text": w["word"].strip()
            })
            global_idx += 1

    return {
        "text": "".join([i['text'] for i in result_aligned["segments"]]),
        "words": output_segments
    }


def batch_transcribe(input_dir, output_dir, model_name, model_weight, device):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]

    feat_extractor = HFASRFeatExtractor(model_name)
    tokenizer = ChineseCharEnglishSpmTokenizer(model_name)
    model = FireRedAsr(model_name, model_weight) if model_weight else FireRedAsr(model_name)

    align_model, metadata = whisperx.load_align_model(
        language_code='zh',
        device=device
    )

    total = len(files)
    success, fail = 0, 0
    print(f"[INFO] Found {total} audio files in '{input_dir}'\n")

    for idx, filename in enumerate(files, start=1):
        print(f"[INFO] ({idx}/{total}) Processing: {filename}")
        input_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        json_path = os.path.join(output_dir, base_name + ".json")

        try:
            result = transcribe_and_align(input_path, model, feat_extractor, tokenizer, align_model, metadata, device)
            with open(json_path, "w", encoding="utf-8") as fout:
                json.dump(result, fout, ensure_ascii=False, indent=2)
            print(f"[SUCCESS] Saved transcription to: {json_path}\n")
            success += 1
        except Exception as e:
            print(f"[ERROR] Failed to process '{filename}': {e}\n")
            fail += 1

    print("=" * 50)
    print(f"[SUMMARY] Completed transcription.")
    print(f"[SUMMARY] Successful: {success} | Failed: {fail}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ASR + WhisperX Alignment Pipeline")
    parser.add_argument("--input_dir", type=str, default="audio", help="Path to input WAV files")
    parser.add_argument("--output_dir", type=str, default="asr_result", help="Path to output JSONs")
    parser.add_argument("--model_name", type=str, default="FireRedASR-AED-L", help="Model name or directory")
    parser.add_argument("--model_weight", type=str, default=None, help="Optional: Path to model weight file")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device to run alignment (auto: select cuda if available)")

    args = parser.parse_args()
    args.device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"

    print(f"[INFO] Using device: {args.device}")
    batch_transcribe(args.input_dir, args.output_dir, args.model_name, args.model_weight, args.device)