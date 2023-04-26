#!/usr/bin/env python3.10

"""
録音を開始する。
loop
    無音を検知する。100ms。
    現在のデータをテキスト化する。
    録音停止でbreak
テキストをGPTに送り込む。(小学生にもわかるようにと補足する)
返ってくる。
gTTSに食わせる。
読み込む。
"""

import pyaudio
import wave
import whisper
import numpy as np
import sys
import pydub
import time
from gtts import gTTS
import tempfile
import json
import openai
from pydub import AudioSegment
from pydub.playback import play
import re
import os

# whisper準備
model = whisper.load_model("large") 

# GPT(API)の準備
creds = json.load( open("./cred.json"))
openai.organization = creds["openai_org"]
openai.api_key      = creds["openai_api_key"]

## 制約
system_messages = []
with open("./system.txt") as fd:
    for line in fd:
        system_messages.append({
            "role": "system", "content": line.strip()
        })

# 録音するパラメータを設定する
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

while True:
    # ストリームを開始する
    print("録音開始")
    # PyAudioオブジェクトを作成する
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    question = ""
    frames   = []
    # 音声を録音する
    """
    録音はすでに開始しておき、有声を検知してから2秒間無音判定が続けば終わり、みたいな。
    """
    windowsz   = RATE ## そりゃ、1秒分しっかりデータを取らないと平均値なんか出ないよ、、 // 100 は 10msくらいしか取れへんからブレて当然やで
    startOnsei = False
    while data := stream.read(windowsz):
        buf    = np.frombuffer(data,dtype=np.int16).flatten().astype(np.float32)
        frames.append(buf)
        print(len(buf))
        print(np.mean(np.abs(buf)))
        if np.mean(np.abs(buf)) < 50.0 and startOnsei:
            if len(frames) > 1 :
                print("処理(dummy)", np.mean(np.abs(buf)))
                ## 正規化
                datanp     = np.frombuffer(b"".join(frames), dtype=np.float32) / 32768.0
                transcribe = model.transcribe(datanp, language="Japanese")
                print("res",transcribe, file=sys.stderr)
                question   = transcribe["text"]
                ## 録音停止
                stream.stop_stream()
                stream.close()
                audio.terminate()
                break
        elif startOnsei == False and np.mean(np.abs(buf)) > 200.0:
            print("集音検知")
            startOnsei = True
    
    if re.search(r"おしまいします。", question ) :
        # gTTSオブジェクトを作成
        tts = gTTS(text="了解です！今日もお疲れ様でした。", lang='ja')
        with tempfile.NamedTemporaryFile(delete=True) as fp:
            tts.save(fp.name)
            play(AudioSegment.from_file(fp.name, format="mp3").speedup(1.1))
            os.exit(0)
    
    message = []
    message += system_messages
    message.append({
        "role": "user", "content": question 
    })
    
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo"
    ,   messages=message
    )
    print(result["usage"], file=sys.stderr)
    finish_reason = result["choices"][0]["finish_reason"]
    print(finish_reason, file=sys.stderr)
    result_message = result["choices"][0]["message"]["content"]
    
    print("------------------------")
    print(result_message)
    print("------------------------")
    
    # gTTSオブジェクトを作成
    tts = gTTS(text=result_message, lang='ja')
    
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts.save(fp.name)
        play(AudioSegment.from_file(fp.name, format="mp3").speedup(1.1))

