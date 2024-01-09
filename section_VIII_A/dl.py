import torch
from torchvision import transforms
import numpy as np
import librosa
from scipy.io import loadmat
import os
import cupy as cp
import cusignal

class reverb_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.audio_folder = os.path.join(dataset_path, 'audio')
        self.noise_folder = os.path.join(dataset_path, 'noise')
        self.reverb_data_folder = 'AIR_1_4/'
        self.audio_list = os.listdir(self.audio_folder)

        self.num_of_audio = len(self.audio_list)

        self.noise_list = os.listdir(self.noise_folder)
        self.num_of_noise = len(self.noise_list)

        self.reverb_data_list = os.listdir(self.reverb_data_folder)
        self.num_of_reverb_data = len(self.reverb_data_list)

        self.nfft = 400  # 25ms: 25*16000/1000
        self.fft_step = 160 # 10ms: 10*16000/1000
        self.overlap = 240 # nfft - overlap

    def __len__(self):
        return self.num_of_audio

    def __getitem__(self,index):
        audio_waveform, fs = librosa.load(os.path.join(self.audio_folder, self.audio_list[index]),sr=None)

        temp1 = 1
        temp2 = 1
        while(temp1 == temp2):
            noise_index = np.random.randint(self.num_of_noise)
            noise_waveform, _ = librosa.load(os.path.join(self.noise_folder, self.noise_list[noise_index]), sr=None)
            clear_noise_spectrogram = cusignal.spectral_analysis.spectral.spectrogram(noise_waveform, fs=fs, window='hann', nperseg=self.nfft, noverlap=self.overlap, nfft=self.nfft, mode='magnitude')
            input_data2 = torch.as_tensor(clear_noise_spectrogram[2][0:200,0:400], device='cuda', dtype=torch.float32)
            temp1 = input_data2.max()
            temp2 = input_data2.min()

        reverb_file_index = np.random.randint(self.num_of_reverb_data, size=2)
        audio_reverb_file = loadmat(os.path.join(self.reverb_data_folder, self.reverb_data_list[reverb_file_index[0]]))
        noise_reverb_file = loadmat(os.path.join(self.reverb_data_folder, self.reverb_data_list[reverb_file_index[1]]))

        audio_reverb_data = audio_reverb_file['h_air'][0]
        noise_reverb_data = noise_reverb_file['h_air'][0]   

        ## Use filter to calculate the reverberation of audio and noise
        reverb_audio_waveform = cusignal.firfilter(audio_reverb_data, audio_waveform)
        reverb_noise_waveform = cusignal.firfilter(noise_reverb_data, noise_waveform)

        ## Normalize
        reverb_audio_waveform = reverb_audio_waveform / reverb_audio_waveform.max()
        reverb_noise_waveform = reverb_noise_waveform / reverb_noise_waveform.max()

        ## mix
        p = np.random.uniform(0,1)

        mix_waveform = p * reverb_audio_waveform + (1-p) * reverb_noise_waveform

        ## To be determined:
        ## Use'psd' or 'magnitude' in spectrogram is in doulbt, but the spectrogram in speaker embedding is using 'psd', so here we temporarily use 'psd'
        mode_str = 'magnitude' # 'magnitude' or 'psd'
        reverb_audio_spectrogram = cusignal.spectral_analysis.spectral.spectrogram(reverb_audio_waveform, fs=fs, window='hann', nperseg = self.nfft, noverlap=self.overlap, nfft=self.nfft, mode=mode_str)

        snr = 10*np.log10((np.sum((p*reverb_audio_waveform)**2))/(np.sum(((1-p)*reverb_noise_waveform)**2)))
        print('p: %f, snr: %f'%(p, snr))

        mix_spectrogram = cusignal.spectral_analysis.spectral.spectrogram(mix_waveform, fs=fs, window='hann', nperseg=self.nfft, noverlap=self.overlap, nfft=self.nfft, mode=mode_str)

        input_data1 = torch.as_tensor(mix_spectrogram[2][0:200,0:400], device='cuda', dtype=torch.float32)
        input_data2 = torch.as_tensor(clear_noise_spectrogram[2][0:200,0:400], device='cuda', dtype=torch.float32)
        output_data = torch.as_tensor(reverb_audio_spectrogram[2][0:200,0:400], device='cuda', dtype=torch.float32)

        input_data1 = (input_data1 - input_data1.min())/(input_data1.max() - input_data1.min())
        input_data2 = (input_data2 - input_data2.min())/(input_data2.max() - input_data2.min())
        output_data = (output_data - output_data.min())/(output_data.max() - output_data.min())

        return torch.cat((input_data1.unsqueeze(0), input_data2.unsqueeze(0)),0), output_data.unsqueeze(0)

class reverb_dataset_test(torch.utils.data.Dataset):
    def __init__(self, dataset_path, p_range=[0,1]):
        self.audio_folder = os.path.join(dataset_path, 'audio')
        self.noise_folder = os.path.join(dataset_path, 'noise')
        self.reverb_data_folder = 'AIR_1_4/'
        self.audio_list = os.listdir(self.audio_folder)

        self.num_of_audio = len(self.audio_list)

        self.noise_list = os.listdir(self.noise_folder)
        self.num_of_noise = len(self.noise_list)

        self.reverb_data_list = os.listdir(self.reverb_data_folder)
        self.num_of_reverb_data = len(self.reverb_data_list)
        self.p_range = p_range

        self.nfft = 400  # 25ms: 25*16000/1000
        self.fft_step = 160 # 10ms: 10*16000/1000
        self.overlap = 240 # nfft - overlap

    def __len__(self):
        return self.num_of_audio

    def __getitem__(self,index):
        audio_waveform, fs = librosa.load(os.path.join(self.audio_folder, self.audio_list[index]),sr=None)

        temp1 = 1
        temp2 = 1
        while(temp1 == temp2):
            noise_index = np.random.randint(self.num_of_noise)
            noise_waveform, _ = librosa.load(os.path.join(self.noise_folder, self.noise_list[noise_index]), sr=None)
            clear_noise_spectrogram = cusignal.spectral_analysis.spectral.spectrogram(noise_waveform, fs=fs, window='hann', nperseg=self.nfft, noverlap=self.overlap, nfft=self.nfft, mode='magnitude')
            input_data2 = torch.as_tensor(clear_noise_spectrogram[2][0:200,0:400], device='cuda', dtype=torch.float32)
            temp1 = input_data2.max()
            temp2 = input_data2.min()

        reverb_file_index = np.random.randint(self.num_of_reverb_data, size=2)
        audio_reverb_file = loadmat(os.path.join(self.reverb_data_folder, self.reverb_data_list[reverb_file_index[0]]))
        noise_reverb_file = loadmat(os.path.join(self.reverb_data_folder, self.reverb_data_list[reverb_file_index[1]]))

        audio_reverb_data = audio_reverb_file['h_air'][0]
        noise_reverb_data = noise_reverb_file['h_air'][0]   

        ## Use filter to calculate the reverberation of audio and noise
        reverb_audio_waveform = cusignal.firfilter(audio_reverb_data, audio_waveform)
        reverb_noise_waveform = cusignal.firfilter(noise_reverb_data, noise_waveform)

        ## Normalize
        reverb_audio_waveform = reverb_audio_waveform / reverb_audio_waveform.max()
        reverb_noise_waveform = reverb_noise_waveform / reverb_noise_waveform.max()

        ## mix
        # p = np.random.uniform(0,1)
        p = np.random.uniform(self.p_range[0], self.p_range[1])

        mix_waveform = p * reverb_audio_waveform + (1-p) * reverb_noise_waveform

        ## To be determined:
        ## Use'psd' or 'magnitude' in spectrogram is in doulbt, but the spectrogram in speaker embedding is using 'psd', so here we temporarily use 'psd'
        mode_str = 'magnitude' # 'magnitude' or 'psd'
        reverb_audio_spectrogram = cusignal.spectral_analysis.spectral.spectrogram(reverb_audio_waveform, fs=fs, window='hann', nperseg = self.nfft, noverlap=self.overlap, nfft=self.nfft, mode=mode_str)
        
        snr = 10*np.log10((np.sum((p*reverb_audio_waveform)**2))/(np.sum(((1-p)*reverb_noise_waveform)**2)))
        print('p: %f, snr: %f'%(p, snr))

        mix_spectrogram = cusignal.spectral_analysis.spectral.spectrogram(mix_waveform, fs=fs, window='hann', nperseg=self.nfft, noverlap=self.overlap, nfft=self.nfft, mode=mode_str)

        input_data1 = torch.as_tensor(mix_spectrogram[2][0:200,0:400], device='cuda', dtype=torch.float32)
        input_data2 = torch.as_tensor(clear_noise_spectrogram[2][0:200,0:400], device='cuda', dtype=torch.float32)
        output_data = torch.as_tensor(reverb_audio_spectrogram[2][0:200,0:400], device='cuda', dtype=torch.float32)

        input_data1 = (input_data1 - input_data1.min())/(input_data1.max() - input_data1.min())
        input_data2 = (input_data2 - input_data2.min())/(input_data2.max() - input_data2.min())
        output_data = (output_data - output_data.min())/(output_data.max() - output_data.min())

        return torch.cat((input_data1.unsqueeze(0), input_data2.unsqueeze(0)),0), output_data.unsqueeze(0)