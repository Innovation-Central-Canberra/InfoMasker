import os
from pathlib import Path
import sys
import librosa
from scipy.io import loadmat
from scipy.signal import lfilter
import numpy as np
import scipy
from PIL import Image

audio_dataset_path = ''
noise_dataset_path = ''
spectrogram_store_path = ''
num_of_data = 60000
audio_people_list = os.listdir(audio_dataset_path)
noise_people_list = os.listdir(noise_dataset_path)

count = 30000
audio_walker = sorted(str(p.stem) for p in Path(audio_dataset_path).glob('*/*/*'+'.wav'))
len_of_audio_walker = len(audio_walker)

# Reverb file path
reverb_file_folder = ''
reverb_file_list = os.listdir(reverb_file_folder)


while (count < num_of_data):
# while (count < 1):
    # Random choose a piece of audio
    idx = np.random.randint(len_of_audio_walker)
    people_id, chapter_id, utterance_id = audio_walker[idx].split('-')
    audio_path = os.path.join(audio_dataset_path, people_id, chapter_id, audio_walker[idx] + '.wav')
    audio_waveform, fs = librosa.load(audio_path, sr=None)

    # Drop the waveform if the length of the waveform is smaller than 5 seconds
    if(len(audio_waveform)/fs <= 5):
        continue
    else:
        start_point = np.random.randint(len(audio_waveform)-5*fs)
        audio_waveform = audio_waveform[start_point : start_point + fs * 5]
    # Random choose two reverb file
    reverb_file_index = np.random.randint(len(reverb_file_list), size=2)
    audio_reverb_file_path = os.path.join(reverb_file_folder, reverb_file_list[reverb_file_index[0]])
    noise_reverb_file_path = os.path.join(reverb_file_folder, reverb_file_list[reverb_file_index[1]])
    audio_reverb_data = loadmat(audio_reverb_file_path)
    noise_reverb_data = loadmat(noise_reverb_file_path)
    # Get the reverb data from dict
    audio_reverb_data = audio_reverb_data['h_air'][0]
    noise_reverb_data = noise_reverb_data['h_air'][0]


    # Random choose a piece of noise
    noise_people_id = noise_people_list[np.random.randint(len(noise_people_list))]
    noise_path = os.path.join(noise_dataset_path, noise_people_id, str(np.random.randint(50))+'.wav')
    noise_waveform, _ = librosa.load(noise_path, sr=None)
    noise_waveform = noise_waveform[0 : 5 * fs]
    
    # Normalize audio and noise waveform 
    audio_waveform = audio_waveform / np.amax(audio_waveform)
    noise_waveform = noise_waveform / np.amax(noise_waveform)

    # Get reverb audio and noise waveform
    reverb_audio_waveform = lfilter(audio_reverb_data, 1, audio_waveform)
    reverb_noise_waveform = lfilter(noise_reverb_data, 1, noise_waveform)
    # Normalize reverb audio and noise waveform
    reverb_audio_waveform = reverb_audio_waveform / np.amax(reverb_audio_waveform)
    reverb_noise_waveform = reverb_noise_waveform / np.amax(reverb_noise_waveform)

    
    # Calculate spectrogram of audio waveform
    audio_spec = np.abs(librosa.stft(reverb_audio_waveform, n_fft=1024, hop_length=512,\
                             window=scipy.signal.windows.hann(1024), center=False))
    audio_spec_uint8 = (((audio_spec - audio_spec.min()) / (audio_spec.max() - audio_spec.min())) * 255.9).astype(np.uint8)
    
    # Calculate spectrogram of nosie waveform
    noise_spec = np.abs(librosa.stft(noise_waveform, n_fft=1024, hop_length=512,\
                                    window=scipy.signal.windows.hann(1024), center=False))
    noise_spec_uint8 = (((noise_spec - noise_spec.min()) / (noise_spec.max() - noise_spec.min())) * 255.9).astype(np.uint8)
#     noise_spec_img = Image.fromarray(noise_spec_uint8[0:256, 0:128])
    
    # Calculate spectrogram of mix waveform
    percent = np.random.randint(1,10) / 10
    mix_waveform = percent * reverb_audio_waveform + (1 - percent) * reverb_noise_waveform
    mix_waveform = mix_waveform / np.amax(mix_waveform)
    mix_spec = np.abs(librosa.stft(mix_waveform, n_fft=1024, hop_length=512,\
                                    window=scipy.signal.windows.hann(1024), center=False))
    mix_spec_uint8 = (((mix_spec - mix_spec.min()) / (mix_spec.max() - mix_spec.min())) * 255.9).astype(np.uint8)
#     mix_spec_img = Image.fromarray(mix_spec_uint8)

    input_image1 = Image.fromarray(mix_spec_uint8[0:256, 0:128])
    input_image2 = Image.fromarray(noise_spec_uint8[0:256, 0:128])
    output_image = Image.fromarray(audio_spec_uint8[0:256, 0:128])
    
    input_image1.save(spectrogram_store_path+'input/mix/'+str(count)+'.png')
    input_image2.save(spectrogram_store_path+'input/noise/'+str(count)+'.png')
    output_image.save(spectrogram_store_path+'output/'+str(count)+'.png')
    
    count = count + 1
    print(count)
