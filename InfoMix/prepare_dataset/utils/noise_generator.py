"""
ToDo:
    1. Noise design for chinese needs improvement
    2. Some textgrids are missed, need to be generated
"""
import os
import numpy as np
from glob import glob
import librosa
import pandas
from ffmpeg import audio as Audio
from uuid import uuid4
import random
import soundfile as sf
from typing import Union
import torchaudio
import torchaudio.transforms as T
import io
from scipy.io import wavfile
import subprocess

MAX_AMP = 0.9


class noise_generator:
    def __init__(
        self,
        target_people_id: str,
        target_subset: Union[str, int],
        dataset_path: str = None,
        textgrid_path: str = None,

    ):
        self.fs = 16000
        self.max_num_of_audio_for_each_user = 100
        self.__target_subset = target_subset


        # Set dataset path and dataset textgrid path
        self.__dataset_path = dataset_path or "/data/shared/speech/LibriSpeech/LibriSpeech"
        self.__textgrid_path = textgrid_path or "/home/huangpeng/data/LibriSpeech/LibriSpeechPhonemeDataForMakingNoise"

        # Set target people id and read phoneme data
        if target_people_id:
            self.set_target_people_id(target_people_id=target_people_id)
        else:
            self.__target_people_id = None
            self.__vowels = None
            self.__consonants = None

    def __read_phoneme_data(self):
        print(f"Reading phoneme data for user: {self.__target_people_id}")
        self.__vowels = []
        self.__consonants = []
        
        speaker_textgrid_path = os.path.join(
            self.__textgrid_path, self.__target_subset, self.__target_people_id
        )
        speaker_textgrid_list = glob(os.path.join(speaker_textgrid_path, '*/*'))
        random.shuffle(speaker_textgrid_list)

        for item in speaker_textgrid_list[: self.max_num_of_audio_for_each_user]:
            audio_path = os.path.join(item, 'file.wav')
            vowel_path = os.path.join(item, "vowel.csv")
            consonant_path = os.path.join(item, 'consonant.csv')

            vowel_time_stamp = pandas.read_csv(vowel_path).values.T[1:3]
            consonant_time_stamp = pandas.read_csv(consonant_path).values.T[1:3]

            audio_wave, current_fs = torchaudio.load(audio_path)
            if int(current_fs) != int(self.fs):
                resampler = T.Resample(
                    current_fs, self.fs, dtype=audio_wave.dtype
                )
                audio_wave = resampler(audio_wave)
            audio_wave = audio_wave.squeeze()

            for i in range(vowel_time_stamp.shape[1]):
                self.__vowels.append(
                    audio_wave[
                        int(vowel_time_stamp[0][i] * self.fs) : int(
                            vowel_time_stamp[1][i] * self.fs
                        )
                    ]
                )
            for i in range(consonant_time_stamp.shape[1]):
                self.__consonants.append(
                    audio_wave[
                        int(consonant_time_stamp[0][i] * self.fs) : int(
                            consonant_time_stamp[1][i] * self.fs
                        )
                    ]
                )

        if len(self.__vowels) == 0:
            print("Error: there is no vowel data for this target people!!!")

    def set_target_people_id(self, target_people_id: str):
        # Check current target people id
        if (
            hasattr(self, "__target_people_id")
            and self.__target_people_id == target_people_id
        ):
            return

        self.__target_people_id = target_people_id
        self.__read_phoneme_data()

    def generate_noise(self, noise_length: int = 10, gap_factor: float = 0.001):
        print(f"Generating noise from target people {self.__target_people_id}'s data")

        noise1 = noise_series1(
            vowel_data=self.__vowels, fs=self.fs, noise_length=noise_length
        )
        noise2 = noise_series2(
            vowel_data=self.__vowels, fs=self.fs, noise_length=noise_length
        )

        noise3 = noise_series3(
            consonant_data=self.__consonants, fs=self.fs, noise_length=noise_length
        )

        # Random weights for each frame (50 ms, 20 frames per second)
        frame_length = 0.05  # 50 ms
        num_of_frame = int(noise_length / frame_length)
        weights = np.random.dirichlet(alpha=np.ones(3), size=num_of_frame).squeeze().T
        weights_1 = weights[0]
        weights_2 = weights[1]
        weights_3 = weights[2]

        # Repeat the weight for each point inter a frame
        num_of_points_per_frame = int(frame_length * self.fs)
        weights_1 = weights_1.repeat(num_of_points_per_frame)
        weights_2 = weights_2.repeat(num_of_points_per_frame)
        weights_3 = weights_3.repeat(num_of_points_per_frame)

        noise = weights_1 * noise1 + weights_2 * noise2 + weights_3 * noise3

        noise = MAX_AMP * noise / np.amax(np.abs(noise))
        return noise


def change_phoneme_speed(phoneme: list, speed_factor: float, fs: int=16000):

    # Format the speed factor
    speed_factor = float(int(speed_factor * 10) / 10)
    assert speed_factor <= 100, "The speech rate change factor is too large!"

    phoneme_int16 = np.int16(phoneme * 32767)
    input_audio_bytes = io.BytesIO()
    wavfile.write(input_audio_bytes, rate=fs, data=phoneme_int16)

    ffmpeg_process = subprocess.Popen(
    [
        "ffmpeg",
        "-i",
        "pipe:O",
        "-filter_complex",
        f"atempo=tempo={speed_factor}",
        "-f",
        "wav",
        "pipe:1",
    ],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    )
    stdout_data, _ = ffmpeg_process.communicate(input=input_audio_bytes.getvalue())
    temp_data = np.frombuffer(buffer=stdout_data, dtype=np.int16)

    # Read output from stdout_data and convert it to float64
    processed_phoneme = np.float64(temp_data/32767).tolist()

    # Remove the first several points because of abnormal part
    processed_phoneme = processed_phoneme[int(fs*0.01):]

    return processed_phoneme
    

class noise_data:
    def __init__(self, size, dtype=float):
        self.data = np.zeros((size), dtype=dtype)
        self.idx = 0
        self.finish = False
        self.max_size = size

    def update(self, input_data):
        if self.idx + len(input_data) > self.max_size:
            self.data[self.idx :] = input_data[: self.max_size - self.idx]
            self.finish = True
            self.idx = self.max_size
        else:
            self.data[self.idx : self.idx + len(input_data)] = input_data
            self.idx += len(input_data)
            if self.idx == self.max_size:
                self.finish = True

    def __len__(self):
        return self.idx


def noise_series1(
    vowel_data: list, fs: int = 16000, noise_length: float = 10, speed_factor=1
):
    num_of_vowel = len(vowel_data)
    noise_max_len = int((noise_length + 4) * fs)
    noise_object = noise_data(noise_max_len)

    while noise_object.finish == False:
        idx = np.random.randint(num_of_vowel)
        phoneme = vowel_data[idx]
        while len(phoneme) < 0.05 * fs:
            idx = np.random.randint(num_of_vowel)
            phoneme = vowel_data[idx]
        noise_object.update(phoneme)

    speed_factor = 1.1
    acc_noise = change_phoneme_speed(noise_object.data, speed_factor, fs)
    assert len(acc_noise) > fs * noise_length, "Error!"

    return acc_noise[: int(fs * noise_length)]


def noise_series2(
    vowel_data: list,
    fs: int = 16000,
    gap_factor: float = 0.005,
    noise_length: float = 10,
):
    num_of_vowel = len(vowel_data)
    noise_max_len = int((noise_length + 4) * fs)
    noise_object = noise_data(noise_max_len)

    # energy_factor_range = [0.5, 2]

    while noise_object.finish == False:
        idx = np.random.randint(num_of_vowel)
        phoneme = vowel_data[idx]
        while len(phoneme) < 0.05 * fs:
            idx = np.random.randint(num_of_vowel)
            phoneme = vowel_data[idx]

        speed_scale_range = [0.5, 1.5]
        augmented_phoneme = change_phoneme_speed(phoneme, random.uniform(speed_scale_range[0], speed_scale_range[1]), fs)
        # augmented_phoneme = phoneme
        # energy_factor = random.uniform(energy_factor_range[0], energy_factor_range[1])
        # augmented_phoneme = augmented_phoneme * energy_factor
        noise_object.update(augmented_phoneme)
        rand_gap = int(np.random.randint(low=0, high=100) * fs * gap_factor)
        noise_object.update([0] * rand_gap)

    # acc_noise = change_phoneme_speed(noise_object.data, 1.1, fs)
    # return acc_noise[:int(fs*noise_length)]

    assert len(noise_object.data) > fs * noise_length, "Error!"

    return noise_object.data[: int(fs * noise_length)]


def noise_series3(consonant_data: list, fs: int = 16000, noise_length: float = 10):
    num_of_consonant = len(consonant_data)
    noise_max_len = int((noise_length + 1) * fs)
    noise_object = noise_data(noise_max_len)

    # return noise_object.data[:int(fs*noise_length)]

    while noise_object.finish == False:
        idx = np.random.randint(num_of_consonant)
        phoneme = consonant_data[idx]
        noise_object.update(phoneme)
    assert len(noise_object.data) > fs * noise_length, "Error!"
    return noise_object.data[: int(fs * noise_length)]
