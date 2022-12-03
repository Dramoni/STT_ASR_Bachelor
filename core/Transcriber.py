import os
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from LiveAudioPreprocessor import LiveAudioPreprocessor


class Transcriber:
    def __init__(self, model_path):
        self.model = self.__load_model(model_path)

    def transcribe(self, wav_path):
        spec, audio, logits, transcriptions = self.__predict_wav(wav_path)
        return transcriptions

    def __load_model(self, model_path):
        print("Loading model...")
        model = K.models.load_model(model_path, custom_objects={'ctc_loss': self.__ctc_loss})
        print(model.summary())
        return model

    def __predict_wav(self, wav_path):
        lap = LiveAudioPreprocessor()
        spec_wav, audio = lap.get_spectrogram_by_wav(wav_path)
        spec = tf.expand_dims(spec_wav, 0)
        pred = self.model.predict(spec)
        pred_logits = np.squeeze(pred, axis=0)
        # print(f"Pred Shape: {pred_logits.shape} ({type(pred_logits)})")

        res = self.__decode_batch_predictions(pred, greedy=True)
        res02 = self.__decode_batch_predictions(pred, greedy=False)
        # print(f"{res}\n{res02}")

        return spec_wav, audio, pred_logits, [res, res02]

    def __decode_batch_predictions(self, pred, greedy=True):
        characters = [x for x in "abcdefghijklmnopqrstuvwxyzäöüß'?! "]
        char_to_num = K.layers.StringLookup(vocabulary=characters, oov_token="")
        num_to_char = K.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

        # print(pred.shape)
        # print('The vocabulary is: {} (size = {})'.format(char_to_num.get_vocabulary(), char_to_num.vocabulary_size()))

        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = K.backend.ctc_decode(pred, input_length=input_len, greedy=greedy, beam_width=10, top_paths=5)

        if not greedy:
            all_outs = []
            res = results[0][0]
            # print(f"res: {res}")
            output_text = []
            for result in res:
                result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
                output_text.append(result)
            all_outs.append(output_text)
            # print(f"tops: {all_outs}")
        else:
            top = results[0][0]
            output_text = []
            for result in top:
                result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
                output_text.append(result)
        return output_text

    def __ctc_loss(self, y_true, y_pred):
        # Compute the training-time loss value
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = K.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss


'''
### EXAMPLE USAGE:

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
transcriber = Transcriber("../saved_models/MASR_Model")
transcriptions = transcriber.transcribe(wav_path="../STT_Inference/test02.wav")
print(transcriptions[0])
'''
