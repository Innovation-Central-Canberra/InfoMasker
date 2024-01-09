import torchaudio.functional as F
import numpy as np
import cusignal

EPS = 1e-6

def resample_audio(audiowave, current_fs, target_fs=8000):
    resampled_waveform = F.resample(audiowave, current_fs, target_fs, lowpass_filter_width=10)
    return resampled_waveform

def normalize_audio(audiowave):
    audiowave = audiowave - np.mean(audiowave)
    audiowave = audiowave / np.amax(np.abs(audiowave))
    return audiowave

def reverb_audio(audiowave, reverb_data):
    origin_length = len(audiowave)
    reverbed_audiowave = cusignal.firfilter(reverb_data, audiowave)
    reverbed_audiowave = normalize_audio(reverbed_audiowave)
    
    # Pad to the same length
    if len(reverbed_audiowave) < origin_length:
        reverbed_audiowave += [0] * (origin_length - len(reverbed_audiowave))

    return reverbed_audiowave[:origin_length]

def mix_to_target_snr(audio,noise,snr=None):
    snr = snr or np.random.uniform(low=-5, high=5)
    assert (snr>=-5) & (snr<=5), 'The target SNR is too high or too low!!!'
    current_snr = calculate_snr(audio, noise)
    factor = np.sqrt((10**(-snr/10))/(10**(-current_snr/10)))
    return audio + factor * noise

def calculate_snr(audio, noise):
    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(audio_power / (noise_power + EPS) + EPS)