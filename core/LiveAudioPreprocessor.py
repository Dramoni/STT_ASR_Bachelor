import itertools
import numpy as np
from scipy.io import wavfile as wav
import tensorflow as tf
from collections import deque
import tensorflow_io as tfio


class LiveAudioPreprocessor:
    def __init__(self, sample_rate=48000, frame_len=256, frame_step=160, fft_len=384):
        self.frame_length = frame_len  # nr of samples
        self.frame_step = frame_step
        self.fft_length = fft_len
        self.avg_loop_time = 0
        self.audio_queue = deque()  # contains nparrays of size = blocksize
        self.samplerate = sample_rate
        # self.stream.start()
        # block size 1 -> 1 frame -> 1 sample

    def audio_callback(self, indata, outdata, frames, time, status):
        # if empty
        # print(indata, frames)       # one frame is one sample, this outputs one discrete value
        self.audio_queue.appendleft(indata.flatten())

    def save_audio_to_wav(self):
        lst = [self.audio_queue.pop() for i in range(len(self.audio_queue))]
        xxx = np.concatenate(lst)
        wav.write("new_test.wav", self.samplerate, xxx)
        return xxx

    def get_spectrogram_by_wav(self, wav_path):
        # self.save_audio_to_wav()
        _file = tf.io.read_file(wav_path)
        audio, _ = tf.audio.decode_wav(_file)
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)

        spectrogram = tf.signal.stft(
            audio, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.fft_length
        )
        print(f"LAP Spec: {spectrogram.shape}")
        # 5. We only need the magnitude, which can be derived by applying tf.abs
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)
        # 6. normalisation
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)
        return spectrogram, audio

    def get_spectrogram_by_mp3(self, mp3_path):
        # self.save_audio_to_wav()
        _file = tf.io.read_file(mp3_path)
        audio = tfio.audio.decode_mp3(_file)
        tensor = tf.cast(audio, tf.float32)
        tensor = tf.squeeze(tensor)
        tensor = tf.ensure_shape(tensor, [None])

        spectrogram = tf.signal.stft(
            tensor, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.fft_length
        )
        # 5. We only need the magnitude, which can be derived by applying tf.abs
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)
        # 6. normalisation
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)
        return spectrogram

    def get_spectrogram_by_numpy(self):
        spectrogram = tf.signal.stft(
            self.get_queue_as_np().astype("float32"), frame_length=self.frame_length
            , frame_step=self.frame_step, fft_length=self.fft_length
        )
        # 5. We only need the magnitude, which can be derived by applying tf.abs
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)
        # 6. normalisation
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)
        return spectrogram

    def get_queue_as_np(self):
        last_x_element_idx = len(self.audio_queue) - self.samplerate * 3
        if last_x_element_idx < 0:
            for i in range(last_x_element_idx * -1):
                self.audio_queue.appendleft(np.zeros(1))
            last_x_element_idx = None
        lst2 = list(itertools.islice(self.audio_queue, last_x_element_idx, None))

        for i in range(self.samplerate * 1):  # remove last second
            self.audio_queue.pop()
        print(len(self.audio_queue))
        return np.concatenate(lst2)

    def get_whole_queue_as_spec(self):
        lst2 = list(itertools.islice(self.audio_queue, None, None))
        spectrogram = tf.signal.stft(
            np.concatenate(lst2).astype("float32"), frame_length=self.frame_length
            , frame_step=self.frame_step, fft_length=self.fft_length
        )
        # 5. We only need the magnitude, which can be derived by applying tf.abs
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)
        # 6. normalisation
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)
        return spectrogram


if __name__ == "__main__":
    pass
