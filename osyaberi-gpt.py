#!/usr/bin/env python3.10

import pyaudio
import wave

##-------------------------------
## 音声の取得

# 録音するパラメータを設定する
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"

# PyAudioオブジェクトを作成する
audio = pyaudio.PyAudio()

# ストリームを開始する
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

print("録音中...")

frames = []

# 音声を録音する
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("録音終了")

# ストリームを停止する
stream.stop_stream()
stream.close()
audio.terminate()

# 録音した音声をファイルに保存する
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

##-------------------------------
## whisperで文字起こし
## TODO: ここもopenaiでもいいかも知れぬ
import whisper

model = whisper.load_model("medium") 
text  = model.transcribe("output.wav")["text"]
print("")
print(text)
print("")

##-------------------------------
## GPTにご連絡
import json
import openai
# TODO: ここを対話にするなら、スレッド感ある感じで返答と質問をまとめて送信する
creds = json.load( open("./osyaberi-gpt.json"))
openai.organization = creds["openai_org"]
openai.api_key      = creds["openai_api_key"]
result = openai.ChatCompletion.create(
    model="gpt-3.5-turbo"
,   messages=[
        {"role":"user", "content" : text }
    ]
)
text   = result["choices"][0]["message"]["content"]
print("")
print(text)
print("")

##-------------------------------
## VOICEVOXで音声に
## TODO: 再生もろとも、テキストを分割してmultiprocessing した方が効率良さそうだなー
## TODO: 音声なんとかしたい
"""
  506  curl -X POST localhost:50021/audio_query?speaker=1 --get --data-urlencode text@text.txt > query.json
  508  curl -X POST -H "content-type: application/json" -d @query.json localhost:50021/synthesis?speaker=1 > audio.wav
"""
import requests 
import urllib

res = requests.post(f"""http://localhost:50021/audio_query?speaker=1&{urllib.parse.urlencode({"text":text})}""")
hdr = {"content-type": "application/json"}
res = requests.post(f"""http://localhost:50021/synthesis?speaker=1""", headers=hdr, json=res.json())
with open("result.wav","wb") as fd:
    fd.write(res.content)

##-------------------------------
## 再生
# 音声ファイルを開く
wf = wave.open('result.wav', 'rb')

# PyAudioオブジェクトを作成する
audio = pyaudio.PyAudio()

# ストリームを開始する
stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

# 音声を再生する
data = wf.readframes(1024)
while data:
    stream.write(data)
    data = wf.readframes(1024)

# ストリームを停止する
stream.stop_stream()
stream.close()
audio.terminate()

