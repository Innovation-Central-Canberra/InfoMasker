import os 
import numpy as np
from glob import glob
import pandas as pd
from scipy import spatial
import soundfile as sf
from tqdm import tqdm
import librosa
import torch
import torchaudio.functional as F
import torchaudio

from utils.rir_dataset import ALL_dataset
from utils.audio_utils import *
from utils.noise_generator import noise_generator
from utils.parameters import *

# 1. For each speaker in the chosen LibriSpeech subsets, find the target speaker who has the similarest voiceprint and then make noise based on the target speakers' audio data.
def create_dataset(store_path:str=None, librispeech_path:str=None, textgrid_path:str=None, people_id_list:list=None):
    if librispeech_path == None:
        librispeech_path = LIBRISPEECH_PATH
    if textgrid_path == None:
        textgrid_path = PHONEME_INDEX_PATH
        
    # Get people list in train-clean-100, train-clean-360, dev-clean, test-clean
    subsets = ['train-clean-100', 'train-clean-360', 'test-clean']
    subset_id_list = {}
    for subset in subsets:
        subset_path = os.path.join(librispeech_path, subset)
        subset_id_list[subset_path] = os.listdir(subset_path)
        if 'test' in subset:
            store_path = TEST_DATASET_PATH
        elif 'train' in subset:
            store_path = TRAIN_DATASET_PATH
        
        # Create audio & noise store path
        audio_path = os.path.join(store_path, 'audio')
        noise_path = os.path.join(store_path, 'noise')
        os.makedirs(audio_path, exist_ok=True)
        os.makedirs(noise_path, exist_ok=True)
            
        # Reading people's voiceprint
        voiceprint_data = np.load('./utils/speaker_voice_embedding/people_embedding.npz')
        people_id_list = voiceprint_data.files
        people_info = pd.read_csv('./utils/speaker_voice_embedding/speakers.csv', engine='python')

        for people_id in tqdm(subset_id_list[subset_path]):
            # Get the speaker's subset
            info_index = list(people_info.index[people_info['id']==int(people_id)])[0]
            speaker_subset = people_info.iloc[info_index]['dataset']
            
            if 'train' not in speaker_subset and 'test' not in speaker_subset:
                continue
            
            temp_distance = np.zeros(len(people_id_list))
            # Get closest people's id
            for i, temp_id in enumerate(people_id_list):
                temp_distance[i] = spatial.distance.cosine(voiceprint_data[str(people_id)], voiceprint_data[str(temp_id)])
            idx = np.argsort(temp_distance)

            # idx[0] is himself, so choose idx[1]
            target_id = people_id_list[idx[1]] 
            print(f'Current people id: {people_id}, Target people id: {target_id}')
            
            # Get closest speaker's subset
            info_index = list(people_info.index[people_info['id']==int(target_id)])[0]
            target_subset = people_info.iloc[info_index]['dataset']

            speaker_audio_list = glob(os.path.join(librispeech_path, speaker_subset, people_id, '*/*.wav'))
            if len(speaker_audio_list) == 0:
                speaker_audio_list = glob(os.path.join(librispeech_path, speaker_subset, people_id, '*/*.flac'))

            # Create noise generator for target speaker
            generator = noise_generator(target_people_id=target_id, 
                                        target_subset=target_subset, 
                                        dataset_path=librispeech_path,
                                        textgrid_path=textgrid_path)
            
            for audio in speaker_audio_list:
                audio_basename = os.path.basename(audio)
                audio_store_path = os.path.join(audio_path, audio_basename).replace('flac', 'wav')
                noise_store_path = os.path.join(noise_path, audio_basename).replace('flac', 'wav')

                if os.path.exists(audio_store_path) & os.path.exists(noise_store_path):
                    continue

                audiowave, sample_rate = sf.read(audio, samplerate=None)
                assert sample_rate == generator.fs, "The audio sample rate is not 16 kHz!!!"
                generated_noise = generator.generate_noise(noise_length=len(audiowave)//sample_rate + 3)

                generated_noise = generated_noise[:len(audiowave)]
                sf.write(noise_store_path, generated_noise, samplerate=sample_rate)
                sf.write(audio_store_path, audiowave, samplerate=sample_rate)
        
# Resample audio and noise to 8kHz
def resample_to_8k():
    target_fs = 8000
    folders = ['audio', 'noise']
    for folder in folders:
        origin_path = os.path.join(TRAIN_DATASET_PATH, folder)
        resampled_path = os.path.join(TRAIN_DATASET_PATH_8k, folder)
        os.makedirs(resampled_path, exist_ok=True)

        audio_list = glob(os.path.join(origin_path, '*.wav'))
        for audio in tqdm(audio_list):
            audioname = os.path.basename(audio)
            target_path = os.path.join(resampled_path, audioname)
            if os.path.exists(target_path):
                continue
            waveform, current_fs = torchaudio.load(audio)
            resampled_waveform = F.resample(waveform, current_fs, target_fs, lowpass_filter_width=10)
            torchaudio.save(target_path, resampled_waveform, target_fs)

        origin_path = os.path.join(TEST_DATASET_PATH, folder)
        resampled_path = os.path.join(TEST_DATASET_PATH_8k, folder)
        os.makedirs(resampled_path, exist_ok=True)

        audio_list = glob(os.path.join(origin_path, '*.wav'))
        for audio in tqdm(audio_list):
            audioname = os.path.basename(audio)
            target_path = os.path.join(resampled_path, audioname)
            if os.path.exists(target_path):
                continue
            waveform, current_fs = torchaudio.load(audio)
            resampled_waveform = F.resample(waveform, current_fs, target_fs, lowpass_filter_width=10)
            torchaudio.save(target_path, resampled_waveform, target_fs)

def create_reverb_mixture(input_folder=TRAIN_DATASET_PATH_8k):
    
    # rename audio folder into audio_origin
    if os.path.exists(os.path.join(input_folder, 'audio')) and not os.path.exists(os.path.join(input_folder, 'origin_audio')):
        a = os.path.join(input_folder, 'audio')
        b = os.path.join(input_folder, 'origin_audio')
        os.system(f'mv {a} {b}')
    if os.path.exists(os.path.join(input_folder, 'noise')) and not os.path.exists(os.path.join(input_folder, 'origin_noise')):
        a = os.path.join(input_folder, 'noise')
        b = os.path.join(input_folder, 'origin_noise')
        os.system(f'mv {a} {b}')
 
    os.makedirs(os.path.join(input_folder, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(input_folder, 'noise'), exist_ok=True)
    os.makedirs(os.path.join(input_folder, 'mixture'), exist_ok=True)

    # Noise reading path
    noise_folder = os.path.join(input_folder, 'origin_noise')

    sample_rate = 8000

    # Initialize rir dataset object
    all_dataset = ALL_dataset(RIR_DATASET_PATH)
    
    skip_count = 0
    audio_list = glob(os.path.join(input_folder,'origin_audio', '*.wav'))
    for audio_path in tqdm(audio_list):
        noise_path = os.path.join(noise_folder, os.path.basename(audio_path))

        mixture_store_path = os.path.join(input_folder, 'mixture', os.path.basename(audio_path))
        audio_store_path = os.path.join(input_folder, 'audio', os.path.basename(audio_path))
        noise_store_path = os.path.join(input_folder, 'noise', os.path.basename(audio_path))

        
        # Check existence
        if os.path.exists(mixture_store_path) and os.path.exists(audio_store_path) and os.path.exists(noise_store_path):
            continue

        try:
            audiowave, fs1 = librosa.load(audio_path, sr=None)
        except:
            print(f'The audio: {audio_path} can not be loaded!')
            skip_count += 1
            continue

        try:
            noisewave, fs2 = librosa.load(noise_path, sr=None)
        except:
            print(f'The noise: {noise_path} can not be loaded!')
            skip_count += 1
            continue

        assert fs1 == sample_rate, "The audio sample rate is not 8 kHz"
        assert fs2 == sample_rate, "The noise sample rate is not 8 kHz"

        audio_rir, rir_info = all_dataset.get_rir_data()
        audio_rir = torch.from_numpy(audio_rir)
        if rir_info['current_fs'] is not sample_rate:
            audio_rir = F.resample(audio_rir, rir_info['current_fs'], sample_rate, lowpass_filter_width=6)

        noise_rir, rir_info = all_dataset.get_rir_data()
        noise_rir = torch.from_numpy(noise_rir)
        if rir_info['current_fs'] is not sample_rate:
            noise_rir = F.resample(noise_rir, rir_info['current_fs'], sample_rate, lowpass_filter_width=6)
           
        reverbed_audiowave = reverb_audio(audiowave, audio_rir)
        reverbed_noisewave = reverb_audio(noisewave, noise_rir)
        mixture = mix_to_target_snr(reverbed_audiowave.get(), reverbed_noisewave.get())

        mixture = normalize_audio(mixture)
        reverbed_audiowave = normalize_audio(reverbed_audiowave.get())

        sf.write(mixture_store_path, mixture, samplerate=sample_rate)
        sf.write(audio_store_path, reverbed_audiowave, samplerate=sample_rate)

        # Random shift origin noise and store
        if len(noisewave) < sample_rate:
            print(f'The noise {noise_path} is shorted than 1 second, skip random shift')
        else:
            shift_length = np.random.randint(low=-sample_rate, high=sample_rate)
            if shift_length > 0:
                noisewave = noisewave[shift_length:]
                noisewave = np.pad(noisewave, (0, shift_length), 'constant', constant_values=0)
            else:
                noisewave = noisewave[:shift_length]
                noisewave = np.pad(noisewave, (0, -shift_length), 'constant', constant_values=0)

        sf.write(noise_store_path, noisewave, samplerate=sample_rate)

    return skip_count

if __name__ == '__main__':
    create_dataset()
    resample_to_8k()
    create_reverb_mixture(input_folder=TRAIN_DATASET_PATH_8k)
    create_reverb_mixture(input_folder=TEST_DATASET_PATH_8k)

    
