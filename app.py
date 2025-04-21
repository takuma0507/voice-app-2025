from flask import Flask, request, jsonify, render_template
import os
import soundfile as sf
import subprocess
from speechbrain.inference import SpeakerRecognition

app = Flask(__name__)

UPLOAD_DIR = 'static/uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

REGISTERED_WEBM = os.path.join(UPLOAD_DIR, 'registered.webm')
REGISTERED_WAV = os.path.join(UPLOAD_DIR, 'registered.wav')
VERIFY_WEBM = os.path.join(UPLOAD_DIR, 'verify.webm')
VERIFY_WAV = os.path.join(UPLOAD_DIR, 'verify.wav')
REGISTER_FLAG = os.path.join(UPLOAD_DIR, 'registered_flag.txt')

# モデルのロード（エラー時もログ出力）
try:
    print("🔄 モデル読み込み中...")
    speaker_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec",
        run_opts={"symlink": False}
    )
    print("✅ モデル読み込み完了！")
except Exception as e:
    print(f"❌ モデルの読み込みに失敗しました: {e}")
    speaker_model = None

# 無音判定関数
def is_silent(wav_path, threshold=0.04):
    data, samplerate = sf.read(wav_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    return max(abs(data)) < threshold

# WebM→WAV変換関数（2秒カット対応）
def webm_to_wav(webm_path, wav_path):
    """
    WebM形式の音声ファイルを2秒で切り出しWAVへ変換
    """
    command = [
        "ffmpeg",
        "-y",
        "-i", webm_path,
        "-t", "2",           # ← 録音長を2秒でカット
        "-ar", "16000",      # ← 16kHzに変換（モデル対応）
        wav_path
    ]
    try:
        subprocess.run(command, check=True)
        print("✅ ffmpegでWAV変換＆2秒切り出し成功")
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg変換エラー: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register_voice', methods=['POST'])
def register_voice():
    audio = request.files['audio_data']
    audio.save(REGISTERED_WEBM)

    try:
        webm_to_wav(REGISTERED_WEBM, REGISTERED_WAV)
    except Exception as e:
        return jsonify({"result": "❌ 音声ファイル変換中にエラーが発生しました。"})

    if is_silent(REGISTERED_WAV):
        os.remove(REGISTERED_WEBM)
        os.remove(REGISTERED_WAV)
        if os.path.exists(REGISTER_FLAG):
            os.remove(REGISTER_FLAG)
        return jsonify({"result": "⚠️ 音声が検出されませんでした。もう一度しっかり発話してください。"})

    with open(REGISTER_FLAG, 'w') as f:
        f.write('registered')

    return jsonify({"result": "✅ 声の登録が完了しました！"})

@app.route('/is_registered', methods=['GET'])
def is_registered():
    registered = os.path.exists(REGISTER_FLAG)
    return jsonify({"registered": registered})

@app.route('/reset_registration', methods=['POST'])
def reset_registration():
    for file_path in [REGISTERED_WEBM, REGISTERED_WAV, REGISTER_FLAG]:
        if os.path.exists(file_path):
            os.remove(file_path)
    return jsonify({"reset": True})

@app.route('/verify_voice', methods=['POST'])
def verify_voice():
    print("🔍 話者判定リクエスト受信")

    if not os.path.exists(REGISTER_FLAG):
        return jsonify({"result": "⚠️ 声がまだ登録されていません。まずは「声を登録🎤」ボタンで登録をお願いします。"})

    audio = request.files['audio_data']
    audio.save(VERIFY_WEBM)
    print("✅ 音声ファイル受信・保存完了")

    try:
        webm_to_wav(VERIFY_WEBM, VERIFY_WAV)
        print("✅ WebM → WAV 変換成功")
    except Exception as e:
        print(f"❌ WebM → WAV 変換失敗: {e}")
        return jsonify({"result": "❌ 音声ファイル変換中にエラーが発生しました。"})

    if is_silent(VERIFY_WAV):
        os.remove(VERIFY_WEBM)
        os.remove(VERIFY_WAV)
        return jsonify({"result": "⚠️ 音声が検出されませんでした。もう一度しっかり発話してください。"})

    if speaker_model is None:
        return jsonify({"result": "❌ モデルが読み込まれていません。再起動が必要です。"})

    print("🧠 モデルでスコア判定開始")
    score, _ = speaker_model.verify_files(REGISTERED_WAV, VERIFY_WAV)
    print(f"✅ スコア計算完了: 類似度スコア = {score.item():.4f}")

    os.remove(VERIFY_WEBM)
    os.remove(VERIFY_WAV)

    SCORE_THRESHOLD = 0.65
    if score.item() >= SCORE_THRESHOLD:
        result = f"✅ 本人の声です。(類似度スコア: {score.item():.2f})"
    else:
        result = f"❌ 本人の声ではありません。(類似度スコア: {score.item():.2f})"

    print(f"🎤 判定結果: {result}")
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run()