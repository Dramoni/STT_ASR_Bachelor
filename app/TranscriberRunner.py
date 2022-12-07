from core.LiveAudioPreprocessor import LiveAudioPreprocessor
import time
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Process


class TranscriberRunner:
    def __init__(self):
        self.lap = LiveAudioPreprocessor()

        self.avg_abs_amplitudes = []
        self.max_recording_seconds = 10.0
        self.validation_frame_size = 1.0
        self.valdation_amplitude_threshold = 0.003

    def listen(self):
        print("[INFO] Recording started")
        self.lap.start_recording()
        t1 = time.perf_counter()
        dt = 0.0
        while dt < self.max_recording_seconds:
            if len(self.lap.audio_queue) > 1_000_000:
                self.lap.stop_recording()
                print("Queue is full!")
                break

            time.sleep(self.validation_frame_size)

            # self.__validate_audio()

            t2 = time.perf_counter()
            dt = t2 - t1
            # print(f"dT: {t2} - {t1} = {dt}")

        self.lap.stop_recording()
        print("[INFO] Recording ended")
        audio = self.lap.save_audio_to_wav()
        print(audio.shape)
        print(len(self.avg_abs_amplitudes))

        plt.plot(audio)
        plt.plot(np.linspace(0, len(audio), len(self.avg_abs_amplitudes)), self.avg_abs_amplitudes, 'o-')
        plt.show()

    def __validate_audio(self):
        audio = self.lap.get_whole_queue_as_np() # macht probleme -> audio stücke fehlen...
        num_samples = self.validation_frame_size * self.lap.samplerate
        self.avg_abs_amplitudes.append((np.abs(audio[int(len(audio) - num_samples):]).mean()))
        last_avg = self.avg_abs_amplitudes[-1]
        print(f"Last Avg: {last_avg}")

tr = TranscriberRunner()
tr.listen()