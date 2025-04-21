# ① import文
from flask import Flask, request, jsonify, render_template
import os
import soundfile as sf
import subprocess
from speechbrain.inference import SpeakerRecognition

app = Flask(__name__)

# ② 環境変数・フォルダ・パス定義
UPLOAD_DIR = 'static/uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

REGISTERED_WEBM = os.path.join(UPLOAD_DIR, 'registered.webm')
REGISTERED_WAV = os.path.join(UPLOAD_DIR, 'registered.wav')
VERIFY_WEBM = os.path.join(UPLOAD_DIR, 'verify.webm')
VERIFY_WAV = os.path.join(UPLOAD_DIR, 'verify.wav')
REGISTER_FLAG = os.path.join(UPLOAD_DIR, 'registered_flag.txt')

speaker_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec",
    run_opts={"symlink": False}
)

# ③ 共通関数（is_silent, webm_to_wav）
def is_silent(wav_path, threshold=0.04):
    data, samplerate = sf.read(wav_path)
    if len(data.shape) > 1:  # ステレオなら
        data = data.mean(axis=1)  # モノラルに変換
    return max(abs(data)) < threshold

def webm_to_wav(webm_path, wav_path):
    command = ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", wav_path]
    subprocess.run(command, check=True)

# ④ Flaskルート関数群
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register_voice', methods=['POST'])
def register_voice():
    audio = request.files['audio_data']
    audio.save(REGISTERED_WEBM)
    webm_to_wav(REGISTERED_WEBM, REGISTERED_WAV)

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
    if not os.path.exists(REGISTER_FLAG):
        return jsonify({"result": "⚠️ 声がまだ登録されていません。まずは「声を登録🎤」ボタンで登録をお願いします。"})

    audio = request.files['audio_data']
    audio.save(VERIFY_WEBM)
    
    webm_to_wav(VERIFY_WEBM, VERIFY_WAV)

    if is_silent(VERIFY_WAV):
        os.remove(VERIFY_WEBM)
        os.remove(VERIFY_WAV)
        return jsonify({"result": "⚠️ 音声が検出されませんでした。もう一度しっかり発話してください。"})

    score, _ = speaker_model.verify_files(REGISTERED_WAV, VERIFY_WAV)

    os.remove(VERIFY_WEBM)
    os.remove(VERIFY_WAV)

    SCORE_THRESHOLD = 0.65
    if score.item() >= SCORE_THRESHOLD:
        result = f"✅ 本人の声です。(類似度スコア: {score.item():.2f})"
    else:
        result = f"❌ 本人の声ではありません。(類似度スコア: {score.item():.2f})"

    return jsonify({"result": result})

# ⑤ 実行部分
if __name__ == "__main__":
    app.run(debug=True)