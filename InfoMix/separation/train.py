"""Recipe for training a neural speech separation system on Libri2/3Mix datasets.
The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer-libri2mix.yaml
> python train.py hparams/sepformer-libri3mix.yaml


The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both libri2mix and
libri3mix.


Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import os
import sys
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
import logging
from torch.utils.tensorboard import SummaryWriter


from utils.separation import Separation

logger = logging.getLogger(__name__)

def dataio_prep(hparams):
    """Creates data processing pipeline"""

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Provide audio pipelines

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav):
        mix_sig = sb.dataio.dataio.read_audio(mix_wav)
        return mix_sig

    @sb.utils.data_pipeline.takes("noise_ref_wav")
    @sb.utils.data_pipeline.provides("noise_ref_sig")
    def audio_pipeline_noise_ref(noise_ref_wav):
        noise_ref_sig = sb.dataio.dataio.read_audio(noise_ref_wav)
        return noise_ref_sig

    @sb.utils.data_pipeline.takes("s1_wav")
    @sb.utils.data_pipeline.provides("s1_sig")
    def audio_pipeline_s1(s1_wav):
        s1_sig = sb.dataio.dataio.read_audio(s1_wav)
        return s1_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_noise_ref)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1)
    
    if (hparams["num_spks"] == 1) and not hparams["use_wham_noise"]:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "noise_ref_sig", "s1_sig"]
        )
    else:
        raise NotImplementedError

    return train_data, valid_data, test_data

if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation
    from prepare_data import prepare_infomix

    run_on_main(
        prepare_infomix,
        kwargs={
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
            "n_spks": hparams["num_spks"],
            "skip_prep": hparams["skip_prep"],
            "fs": hparams["sample_rate"],
        },
    )

    # Create dataset objects
    if hparams["dynamic_mixing"]:
        raise NotImplementedError
        # from dynamic_mixing import (
        #     dynamic_mix_data_prep_infomix as dynamic_mix_data_prep,
        # )

        # dm_hparams = {
        #     "train_data": hparams["train_data"],
        #     "data_folder": hparams["data_folder"],
        #     "base_folder_dm": hparams["base_folder_dm"],
        #     "sample_rate": hparams["sample_rate"],
        #     "num_spks": hparams["num_spks"],
        #     "training_signal_len": hparams["training_signal_len"],
        #     "dataloader_opts": hparams["dataloader_opts"],
        # }

        # train_data = dynamic_mix_data_prep(dm_hparams)
        # _, valid_data, test_data = dataio_prep(hparams)
    else:
        train_data, valid_data, test_data = dataio_prep(hparams)

    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected(
            device=run_opts["device"]
        )

    # For tensorboard 
    log_directory = os.path.join(hparams['save_folder'], 'tensorboard')
    tb = SummaryWriter(log_directory)

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        tb=tb
    )

    # re-initialize the parameters if we don't use a pretrained model
    if "pretrained_separator" not in hparams:
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    if not hparams["test_only"]:
        # Training
        separator.fit(
            separator.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )

    # Eval
    separator.evaluate(test_data, min_key="si-snr")
    separator.save_results(test_data)
