# Style-Bert-VITS2-Mac — TTS Server 改変概要

## 概要

Style-Bert-VITS2 の日本語音声合成サーバーを macOS（Apple Silicon）で安定して動作させるための改変です。

## 主な改変内容

### 1. BERT 特徴量の dtype 不整合修正（`style_bert_vits2/nlp/japanese/bert_feature.py`）

DeBERTa-v2 BERT モデルの hidden_states は float16 で出力されますが、音声合成モデル（JP-Extra 系）の TextEncoder は float32 を期待しています。この dtype の不一致により `RuntimeError: Input type (c10::Half) and bias type (float) should be the same` が発生していました。

**修正**: `extract_bert_feature()` において、CPU 転送後に明示的に `.float()` を呼び出し、float32 に変換します。

- Line 48: `...[0].cpu()` → `...[0].cpu().float()`
- Line 54: `...[0].cpu()` → `...[0].cpu().float()`

### 2. MPS 実行時のフォールバック（`server_speech_fastapi.py`）

macOS 26.5 / Apple T6041 において PyTorch 2.7 の MPS backend に `RuntimeError: invalid low watermark ratio 1.4` のバグが存在し、MPS 上の全演算がブロックされます。

**修正**: サーバー起動時に MPS の動作を単一テンソル演算で試し、失敗した場合に CPU にフォールバックします。

### 3. 依存関係の調整（`requirements.txt`）

- `torch==2.5.1`, `torchaudio==2.5.1` に固定（CVE-2025-32434 対策 + 現在硬件の安定動作版）

## テスト手順

### 1. サーバー起動

```bash
.venv/bin/python server_speech_fastapi.py
```

起動メッセージが確認できます:
```
INFO  | server_speech_fastapi.py:355 | server listen: http://127.0.0.1:9000
INFO  | server_speech_fastapi.py:356 | API docs: http://127.0.0.1:9000/docs
```

正常に MPS が使えない場合は以下の警告が表示されます:
```
WARNING | server_speech_fastapi.py:118 | MPS test failed (watermark ratio bug). Falling back to CPU.
```

### 2. TTS リクエスト（curl）

日本語テキスト "こんにちは" でテスト:

```bash
curl "http://127.0.0.1:9000/voice?text=%E3%81%93%E3%82%93%E3%81%AB%E3%81%A1%E3%81%AF&speaker_id=0&model_id=0" -o output.wav
```

成功時のレスポンス: `200 OK`、ファイルサイズは約 30KB 前後の WAV ファイルが生成されます。

### 3. 動作確認

```bash
file output.wav
ffprobe output.wav
```

以下のプロパティが確認できるはずです:
- 形式: WAV (RIFF)
- サンプルレート: 44100 Hz
- チャネル: 1 (モノラル)
- 形式: s16 (16-bit PCM)

### 4. ログの確認

サーバー起動中に以下の DEBUG ラインが表示され、全 dtype が `torch.float32` となっていれば修正成功です:

```
DEBUG JP-Extra TextEncoder.forward: bert input dtype=torch.float32, bert_proj.weight dtype=torch.float32
```

## 既知の問題

- **MPS 非対応**: watermark ratio バグのため、現時点では CPU フォールバックが必要です
- **日本語テキストの長さ制限**: 既定で 100 文字（`config.yml` の `server.limit` で変更可能）
