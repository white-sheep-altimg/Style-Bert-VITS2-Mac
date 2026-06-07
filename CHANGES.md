# Change Log — Style-Bert-VITS2-Mac TTS Server

## 改変一覧

### [Fix] BERT 特徴量の dtype 不整合による JP-Extra 合成エラー

**ファイル**: `style_bert_vits2/nlp/japanese/bert_feature.py`
**行**: 48, 54
**状態**: 修正済み

#### 発現事象

JP-Extra 系モデル（JP-NVJ, JP-Nova 等）の音声合成時に以下エラーで晶霊停止:

```
RuntimeError: Input type (c10::Half) and bias type (float) should be the same
```

エラー発生箇所: `TextEncoder.forward()` 内、`bert_proj` (Conv1d) の計算時。

#### 原因解析

推論パイプラインのデータフロー:

```
extract_bert_feature() → get_text() → infer() → SynthesizerTrnJPExtra.infer() → TextEncoder.forward()
```

問題の発生箇所は `extract_bert_feature()` の48行目:

```python
# 問題のコード
res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
```

DeBERTa-v2-large-japanese-char-wwm の hidden_states は `torch.float16` で出力されます。`.cpu()` 演算は dtype を変換しないため、この時点で生成されるテンソルは引き続き `float16` です。

一方、JP-Extra モデルの `TextEncoder.bert_proj` は `torch.float32` で初期化されています（ safetensors ファイルからロードされる重み）。Conv1d 演算では入力テンソルと重み/バイアスの dtype が一致している必要があり、`float16` 入力と `float32` 重みの組み合わせが `RuntimeError` を引き起こします。

#### 検証方法

各推論ステージに `print(f"DEBUG ... dtype=...")` を挿入し、テンソルの dtype を追跡:

```
DEBUG extract_bert_feature return: dtype=torch.float16, shape=torch.Size([1024, 23])  ← 問題発生源
DEBUG get_text return: bert dtype=torch.float32, ja_bert dtype=torch.float16, en_bert dtype=torch.float32
DEBUG infer passing to SynthesizerTrn: ja_bert dtype=torch.float16, shape=torch.Size([1, 1024, 23])
DEBUG SynthesizerTrnJPExtra.infer entry: bert dtype=torch.float16, shape=torch.Size([1, 1024, 23])
DEBUG JP-Extra TextEncoder.forward: bert input dtype=torch.float16, bert_proj.weight dtype=torch.float32
```

この追跡により、`float16` が `extract_bert_feature()` の返り値で既に混入していることが conclusively に証明されました。`get_text()` や `infer()` 内の `.to(device)` は dtype を変換しないため、問題の発生源は `extract_bert_feature()` であることが明確になります。

#### 修正内容

CPU 転送後に明示的に `.float()` を呼び出し、`float32` に変換:

```python
# BEFORE (line 48):
res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

# AFTER:
res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu().float()
```

補助テキスト（assist_text）側の同じ処理（54行目）も同様に修正:

```python
# BEFORE (line 54):
style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()

# AFTER:
style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu().float()
```

#### 修正後の動作

```
DEBUG JP-Extra TextEncoder.forward: bert input dtype=torch.float32, bert_proj.weight dtype=torch.float32, bert_proj.bias dtype=torch.float32
INFO  | Audio data generated successfully
SUCCESS | Audio data generated and sent successfully
```

全 dtype が `torch.float32` に統一され、音声合成が正常に完了します。

---

### [Fix] PyTorch MPS watermark ratio バグによるサーバー起動ブロック

**ファイル**: `server_speech_fastapi.py`
**行**: 104-121
**状態**: 修正済み（CPU フォールバック実装）

#### 発現事象

macOS 26.5 + Apple T6041 + PyTorch 2.7.1 で、サーバー起動時に即時クラッシュ:

```
RuntimeError: invalid low watermark ratio 1.4
```

このエラーは MPS 上の基本テンソル演算（`torch.randn(1, 1, device="mps")`）でも再現する系统性バグです。

#### 原因

PyTorch 2.4-2.7 で Apple Silicon の MPS backend に `watermark_ratio` の制御バグが存在。macOS 26.5 (kernel 25.x) + T6041 の組み合わせで特に顕著。

#### 修正内容

MPS 利用前に単一テンソル演算で動作試験を行い、失敗時に CPU にフォールバック:

```python
# 使えれば CUDA、次に MPS を試す、ダメなら CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_built():
    _test_mps = False
    try:
        torch.randn(1, 1, device="mps")
        _test_mps = True
    except RuntimeError:
        _test_mps = False
    if _test_mps:
        device = "mps"
    else:
        logger.warning("MPS test failed (watermark ratio bug). Falling back to CPU.")
        device = "cpu"
else:
    device = "cpu"

# CPU なら、マシンのキャパの半分を割り振る
if device == "cpu":
    torch.set_num_threads(os.cpu_count()//2)
```

---

### [Adjust] 依存関係の調整

**ファイル**: `requirements.txt`
**状態**: 修正済み

#### 変更内容

```diff
-torch==2.7.1
-torchaudio==2.7.1
+torch==2.6.0
+torchaudio==2.6.0
```

#### 理由

1. **CVE-2025-32434**: PyTorch < 2.6 は `torch.load()` の脆弱性により transformers がブロックする
2. **Hardware stability**: 現在の硬件（T6041 / macOS 26.5）では 2.6.0 で安定動作を確認
3. **MPS fallback により 過剰な依存回避**: MPS を使わないため、2.6.0 を採用

---

---

### [Fix] transcribe.py の macOS 対応（GPU なし環境での文字起こし対応）

**ファイル**: `transcribe.py`
**状態**: 修正済み

#### 変更内容

1. `--device` 引数のデフォルト値を `"cuda"` から `"auto"` に変更
2. `auto` モードで macOS の MPS 利用可否を自動検出（利用不可なら CPU フォールバック）
3. `transcribe_files_with_hf_whisper()` 内で HF Whisper pipeline の `device` パラメータがハードコード (`"cuda"`) されていたのを、引数 `device` で受け取るように修正
4. `__main__` ブロックで `torch` をインポート可能にするためファイル先頭に `import torch` を追加

---

### [Fix] style_gen.py の依存パッケージ互換性パッチ

**ファイル**: `style_gen.py`
**状態**: 修正済み

#### 発現事象

学習パイプラインの `style_gen` ステップで以下エラーが発生し、音声スタイル特徴量の生成が失敗:

```
TypeError: hf_hub_download() got an unexpected keyword argument 'use_auth_token'
_pickle.UnpicklingError: Weights only load failed... PyTorch 2.6 changed default of weights_only from False to True
```

#### 原因

1. `huggingface_hub 1.17.0` で `use_auth_token` 引数が削除され `token` に変更されたが、`pyannote/audio 3.4.0` が古い引数名で呼び出している
2. `torch 2.6+` で `torch.load()` の `weights_only` 引数のデフォルトが `True` に変わったが、`pyannote/audio` のチェックポイントは古い pickle フォーマットで `weights_only=False` でのみロード可能

#### 修正内容

3つのパッチを `style_gen.py` 先頭に追加:
- `torch.load` のラッパー: `weights_only=False` を強制
- `lightning_fabric._load` のラッパー: `weights_only=False` を強制
- `hf_hub_download` のラッパー: `use_auth_token` を `token` に変換

---

### [Debug] トレースコード追加（開発時一時的）

**ファイル**: `style_bert_vits2/nlp/japanese/bert_feature.py`
**行**: 74
**状態**: 保持（将来的に削除可能）

```python
print(f"DEBUG extract_bert_feature return: dtype={result.dtype}, shape={result.shape}", flush=True)
```

推論パイプラインの可観測性を高めるための debug print。正常動作確認後は削除可能です。
