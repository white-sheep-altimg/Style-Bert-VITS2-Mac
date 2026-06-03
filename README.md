# Style-Bert-VITS2-Mac — TTS Server 改変概要

## 概要

Style-Bert-VITS2 の日本語音声合成サーバーを macOS（Apple Silicon）で安定して動作させるための改変です。

## 主な改変内容

### 1. BERT 特徴量の dtype 不整合修正（`style_bert_vits2/nlp/japanese/bert_feature.py`）

DeBERTa-v2 BERT モデルの hidden_states は float16 で出力されますが、音声合成モデル（JP-Extra 系）の TextEncoder は float32 を期待しています。この dtype の不一致により `RuntimeError: Input type (c10::Half) and bias type (float) should be the same` が発生していました。

**修正**: `extract_bert_feature()` において、CPU 転送後に明示的に `.float()` を呼び出し、float32 に変換します。

- Line 48: `...[0].cpu()` → `...[0].cpu().float()`
- Line 54: `...[0].cpu()` → `...[0].cpu().float()`

### 2. MPS 実行時のフォールバック（`server_speech_fastapi.py`, `app.py`）

macOS 26.5 / Apple Silicon において PyTorch 2.7 の MPS backend に `RuntimeError: invalid low watermark ratio 1.4` のバグが存在し、`nn.Module.to("mps")` を含む全 MPS 上演算がブロックされます。

**修正**: `app.py` および `server_speech_fastapi.py` でデバイス選択時に MPS を使用せず、常に CPU を強制使用する。`infer.py` にも安全策として `_to_device()` ラッパーを追加し、万が一 MPS が指定された場合の low watermark エラーをキャッチして CPU にフォールバックする。

### 3. 依存関係の調整（`requirements.txt`）

- `torch==2.7.1`, `torchaudio==2.7.1` に固定（CVE-2025-32434 対策 + macOS Apple Silicon 対応）
- `numpy<2`（既存のコードとの互換性のため）

### 4. 新規インストール時の必須手順

新規clone後、以下の依存パッケージを追加インストールする必要があります:

```bash
# pyenvまたはpython3.11でvenvを作成
python3.11 -m venv .venv
source .venv/bin/activate

# 依存パッケージをインストール
pip install -r requirements.txt

# setuptoolsのpkg_resources使用のため（librosa依存）
pip install 'setuptools<81'

# transformersのaudio_utilsがsoxrを必要とするため
pip install soxr

# モデルのダウンロード
python initialize.py
```

### 5. Pythonバージョンの固定

`.python-version` ファイルに `3.11.13` を記載し、pyenvでPython 3.11 系を使用します。

### 6. app.py のデバイス選択変更（`app.py`）

Gradio WebUI（`app.py`）でも同様に MPS を使用せず CPU を強制する。環境変数 `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` での回避も試みたが、gradio の import 時に MPS バックエンドが初期化されてしまうため機能しなかった。CPU スレッド数は `os.cpu_count() // 2` を割り当てる。

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
- **推論速度**: CPU 実行のため、MPS が正常動作すれば GPU 加速能达到レベルには達しません。ただし実用上は問題ない速度です
- **日本語テキストの長さ制限**: 既定で 100 文字（`config.yml` の `server.limit` で変更可能）
