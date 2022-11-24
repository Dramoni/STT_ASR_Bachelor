import threading
import time
import os
import webbrowser
import librosa
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from pydub.playback import play
from tensorflow.keras.models import load_model

class WakeWordDetector:
    def __init__(self, model_path=r"E:\Bachelor\saved_models\wake_word_conv2d_aug", response_path=r"E:\Bachelor\GLaDOS_Okay.wav"):
        self.sample_rate = 22050
        self.seconds = 2
        self.n_mfcc = 20

        self.okay_audio = AudioSegment.from_wav(response_path)

        print("loading model...")
        self.model = load_model(model_path)
        print(self.model.summary())
        print("WakeWordDetector initialized!")

    def listener(self):
        print("listening now...")
        while True:
            rec = sd.rec(int(self.seconds * self.sample_rate), samplerate=self.sample_rate, channels=1)
            sd.wait()
            mfcc = librosa.feature.mfcc(y=rec.ravel(), sr=self.sample_rate, n_mfcc=self.n_mfcc)
            mfcc_processed = mfcc # np.mean(mfcc.T, axis=0)
            self.prediction_thread(mfcc_processed)
            time.sleep(0.001)

    def voice_thread(self):
        listen_thread = threading.Thread(target=self.listener, name="ListeningFunction")
        listen_thread.start()

    def prediction(self, y):
        pred = self.model.predict(np.expand_dims(y, axis=0))
        if pred[:, 1] > 0.95:
            print("WAKE WORD DETECTED!")
            print(f"Confidence: {pred[:, 1]}")
            # os.startfile(r"E:\Bachelor")
            play(self.okay_audio)
            webbrowser.open("http://localhost/STT")
        else:
            print("No wake word...")
            print(f"Confidence: {pred[:, 0]}")
        time.sleep(0.1)

    def prediction_thread(self, y):
        pred_thread = threading.Thread(target=self.prediction, name="PredictFunction", args=(y,))
        pred_thread.start()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
wwd = WakeWordDetector()
wwd.voice_thread()