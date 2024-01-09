import os
import sys
sys.path.append('/home/huangpeng/code/infomasker_experiments/denoising_for_recording')
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
import torchaudio.functional as F
import torchaudio
from glob import glob
from tqdm import tqdm

from utils.separation import Separation

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def enhance_folder(separator, input_folder:str, output_folder:str):
    mixture_list = glob(os.path.join(input_folder, 'mixture', '*.wav'))

    for mixture in tqdm(mixture_list):
        noise_ref = mixture.replace('mixture', 'noise')
        store_path = os.path.join(output_folder, os.path.basename(mixture))
        result = separator.separate_file(noisy_audio_path=mixture, noise_ref_path=noise_ref)
        result = result[:,:,0].detach()
        torchaudio.save(store_path, result, 8000)

if __name__ == '__main__':
    
    # Load hyperparameters file with command-line overrides
    hparams_file = 'hparams/sepformer-infonet.yaml'
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)
    
    hparams['evaluate_pretrained_separator'].collect_files()
    hparams["evaluate_pretrained_separator"].load_collected(device="cuda")

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"]
    )
    separator.zero_grad()
    # a = separator.separate_file(noisy_audio_path='mixture.wav', noise_ref_path='noise_ref.wav')
    # result = a[:,:,0].detach()
    # torchaudio.save('result.wav', result, 8000)
    
    for snr in range(-9, 6, 1):
        input_folder = f'/home/huangpeng/code/infomasker_experiments/denoising_for_recording/infonet/data/real_world_test_data/test_data/snr_{snr}'
        output_folder = f'/home/huangpeng/code/infomasker_experiments/denoising_for_recording/infonet/data/denoised_real_world_test_data/snr_{snr}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        enhance_folder(separator, input_folder=input_folder, output_folder=output_folder)

    
    


            
