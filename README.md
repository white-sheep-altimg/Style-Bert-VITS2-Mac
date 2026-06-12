# Style-Bert-VITS2-Mac — TTS Server 改変概要

## 概要

Style-Bert-VITS2 の日本語音声合成サーバーを macOS（Apple Silicon）で安定して動作させるための改変です。
tokyohandsome様の https://github.com/tokyohandsome/Style-Bert-VITS2-Mac.git のフォークです。

これは個人的なハック＋Claude-Code利用による改修です。自身の環境でしかテストしていませんので問題が残っているかもしれません。
オリジナル，フォーク元ともにCLI起動を前提としておりますが，サービスとしての起動コマンドも追加しています。

音声合成，学習など一通りは動かしていますが，ほぼ全ての機能をCPUにフォールバクして動かしています。PyTorchなど依存関係の問題もあるMPSでの動作は難しいかもしれません。

なお，ライセンスはフォーク元，オリジナルを暗唱してください。


## サービス管理（`hsat`）

API サーバー（`server_speech_fastapi.py`）を macOS 上でバックグラウンドサービスとして管理できます。

```bash
./hsat start          # バックグラウンド起動
./hsat start --cpu    # CPU モードで起動
./hsat stop           # サーバーを停止
./hsat status         # 起動状態を確認
./hsat logs           # ログをリアルタイム表示（tail -f）
```

- ログは `logs/server-YYYYMMDD-HHMMSS.log` に自動保存
- 重複起動ガード機能付き
- ターミナルを閉じてもサーバーは継続して動作

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

### 7. 文字起こし（`transcribe.py`）の macOS 対応

`--device` のデフォルト値を `"cuda"` から `"auto"` に変更。`auto` モードでは macOS で MPS が利用可能な場合は MPS を、否则 CPU を自動選択する。また HF Whisper pipeline の `device` パラメータがハードコードされていたのを引数で受け取るように修正。

### 8. 学習パイプラインの依存パッケージ互換性パッチ（`style_gen.py`）

PyTorch 2.6+ で `torch.load()` の `weights_only` 引数のデフォルトが `True` に変わったことに起因する古いチェックポイントのロード失敗をパッチで回避。また `huggingface_hub >= 0.24` で `use_auth_token` が `token` に変更されたことに伴う `pyannote/audio` の互換性問題もパッチで修正。

## テスト手順

### 1. サーバー起動

`hsat` コマンドでバックグラウンド起動（推奨）:

```bash
./hsat start
```

または直接起動:

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

`hsat logs` でバックグラウンドログをリアルタイム表示:

```bash
./hsat logs
```

サーバー起動中に以下の DEBUG ラインが表示され、全 dtype が `torch.float32` となっていれば修正成功です:

```
DEBUG JP-Extra TextEncoder.forward: bert input dtype=torch.float32, bert_proj.weight dtype=torch.float32
```

## 既知の問題

- **MPS 非対応**: watermark ratio バグのため、現時点では CPU フォールバックが必要です
- **推論速度**: CPU 実行のため、MPS が正常動作すれば GPU 加速能达到レベルには達しません。ただし実用上は問題ない速度です
- **日本語テキストの長さ制限**: 既定で 100 文字（`config.yml` の `server.limit` で変更可能）



# Original README.md
フォーク元のREADME.mdはREADME_orig.mdです。

tokyohandsome様がハックしていなければTahoeで動かそうという気持ちには，そもそもならなかったでしょう。
とても感謝しております。

https://github.com/tokyohandsome/Style-Bert-VITS2-Mac.git

