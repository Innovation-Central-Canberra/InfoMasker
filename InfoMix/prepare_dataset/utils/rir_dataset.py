import os
from enum import Enum
from glob import glob
from scipy.io import loadmat
import numpy as np
import subprocess
import soundfile as sf

available_dataset = ["AIR_1_4", "SPARGAIR", "RWCP"]

class Engine(object):
    def __init__(
        self,
        download_link,
        dataset_base_folder,
        current_fs,
        download_filename,
        dataset_name,
    ):
        # super(self).__init__()
        self.download_link = download_link
        self.dataset_base_folder = dataset_base_folder
        self.current_fs = current_fs
        self.download_filename = download_filename
        self.dataset_name = dataset_name
        self._check_exists()
        self._generate_rir_file_list()

    def get_rir_data(relative_path: str = None):
        """
        Inputs:
            Relative path: relative path of the rir_file
        Returns:
            rir_data (n,): rir data
            rir_info (dict): sample rate of the rir data and the rir data filename
        rir_info = {'current_fs': self.current_fs}
        """
        raise NotImplementedError

    def _generate_rir_file_list(self):
        raise NotImplementedError

    @classmethod
    def create(cls, x, **kwargs):
        if x == 'AIR_1_4':
            return AIR_1_4_dataset(**kwargs)
        elif x == 'SPARGAIR':
            return SPARGAIR_dataset(**kwargs)
        elif x == 'RWCP':
            return RWCP_dataset(**kwargs)
        else:
            raise ValueError(f"Cannot create {cls.__name__} of type '{x}'")

    def __len__(self):
        return len(self.rir_file_list)

    def info(self):
        print(f"Dataset Name: {self.dataset_name}")
        print(f"Number of RIR data in dataset: {len(self.rir_file_list)}")
        print(f"Sample Rate: {self.current_fs}")

    def _check_exists(self):
        """
        Check the existence of the target dataset. If not, download and unzip it.
        """
        if os.path.exists(self.dataset_base_folder):
            return
        elif not os.path.exists(os.path.join(os.path.dirname(self.dataset_base_folder),self.download_filename)):
            # Download
            print(
                f"Downloading the dataset:{os.path.basename(self.dataset_base_folder)}"
            )
            download_command = [
                "wget",
                "-P",
                os.path.dirname(self.dataset_base_folder),
                self.download_link,
            ]
            subprocess.run(download_command)
        # Unzip
        if '.zip' in self.download_filename:
            unzip_command = [
                "unzip",
                os.path.join(os.path.dirname(self.dataset_base_folder),self.download_filename),
                "-d",
                os.path.dirname(self.dataset_base_folder),
            ]
        else:
            unzip_command = [
                "tar",
                "-xvzf",
                os.path.join(os.path.dirname(self.dataset_base_folder),self.download_filename),
                "--directory",
                os.path.dirname(self.dataset_base_folder),
            ]

        subprocess.run(unzip_command)

class SPARGAIR_dataset(Engine):
    """
    Create an RIR dataset object based on METU SPARG Eigenmike em32 Acoustic Impulse Response Dataset v0.1.0 (https://zenodo.org/record/2635758)
    """

    def __init__(self, **kwargs):
        download_link = "https://zenodo.org/record/2635758/files/spargair.zip"
        download_filename = "spargair.zip"
        current_fs = 48000

        if "dataset_base_folder" in kwargs.keys():
            dataset_base_folder = kwargs["dataset_base_folder"]
        elif "base_folder" in kwargs.keys():
            dataset_base_folder = os.path.join(
                kwargs["base_folder"], "spargair"
            )

        super().__init__(
            download_link=download_link,
            dataset_base_folder=dataset_base_folder,
            current_fs=current_fs,
            download_filename=download_filename,
            dataset_name="SPARGAIR",
        )

    def get_rir_data(self, relative_path: str = None):
        rir_info = {"current_fs": self.current_fs, "dataset_name": self.dataset_name}

        # if relative_path is None: randomly get
        if relative_path == None:
            rir_file_path = np.random.choice(self.rir_file_list, size=1)[0]
            rir_data, _ = sf.read(rir_file_path, samplerate=None)
            sep_path = rir_file_path.split(os.sep)
            rir_info["rir_filename"] = os.path.join(*sep_path[-3:])
        else:
            rir_file_path = os.path.join(
                self.dataset_base_folder, relative_path.strip(os.sep)
            )
            rir_data, _ = sf.read(rir_file_path, samplerate=None)
            sep_path = rir_file_path.split(os.sep)
            rir_info["rir_filename"] = relative_path

        return rir_data, rir_info

    def _generate_rir_file_list(self):
        self.rir_file_list = glob(os.path.join(self.dataset_base_folder, "*/*/*.wav"))

class AIR_1_4_dataset(Engine):
    def __init__(self, **kwargs):
        """
        Create an RIR dataset object based on AIR_1_4 dataset (https://www.iks.rwth-aachen.de/forschung/tools-downloads/databases/aachen-impulse-response-database/)

        """
        download_link = "https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/forschung/tools-downloads/air_database_release_1_4.zip"
        download_filename = "air_database_release_1_4.zip"
        current_fs = 48000

        if "dataset_base_folder" in kwargs.keys():
            dataset_base_folder = kwargs["dataset_base_folder"]
        elif "base_folder" in kwargs.keys():
            dataset_base_folder = os.path.join(
                kwargs["base_folder"], "AIR_1_4"
            )

        super().__init__(
            download_link=download_link,
            dataset_base_folder=dataset_base_folder,
            current_fs=current_fs,
            download_filename=download_filename,
            dataset_name="AIR_1_4",
        )

    def get_rir_data(self, relative_path: str = None):
        rir_info = {"current_fs": self.current_fs, "dataset_name": self.dataset_name}
        # if relative_path is None: randomly get
        if relative_path == None:
            mat_file_path = np.random.choice(self.rir_file_list, size=1)[0]
            rir_data = loadmat(mat_file_path)["h_air"][0]
            rir_info["rir_filename"] = os.path.basename(mat_file_path)
        else:
            mat_file_path = os.path.join(self.dataset_base_folder, relative_path)
            rir_data = loadmat(mat_file_path)["h_air"][0]
            rir_info["rir_filename"] = kwargs["rir_filename"]
        return np.array(rir_data), rir_info

    def _generate_rir_file_list(self):
        self.rir_file_list = glob(os.path.join(self.dataset_base_folder, "*.mat"))

class RWCP_dataset(Engine):
    def __init__(self, **kwargs):
        """
        Create an RIR dataset object based on RWCP dataset (https://us.openslr.org/resources/13/RWCP.tar.gz)

        """
        download_link = "https://us.openslr.org/resources/13/RWCP.tar.gz"
        download_filename = "RWCP.tar.gz"
        current_fs = 48000
        self.dtype = "float32"

        if "dataset_base_folder" in kwargs.keys():
            dataset_base_folder = kwargs["dataset_base_folder"]
        elif "base_folder" in kwargs.keys():
            dataset_base_folder = os.path.join(
                kwargs["base_folder"], "RWCP"
            )

        super().__init__(
            download_link=download_link,
            dataset_base_folder=dataset_base_folder,
            current_fs=current_fs,
            download_filename=download_filename,
            dataset_name="RWCP",
        )

    def get_rir_data(self, relative_path:str=None):
        rir_info = {"current_fs": self.current_fs, "dataset_name": self.dataset_name}

        if relative_path == None:
            rir_file_path = np.random.choice(self.rir_file_list, size=1)[0]
            rir_data = np.memmap(rir_file_path, self.dtype, "r").reshape((1, -1))
            rir_data = np.array(rir_data.tolist())
            rir_info['rir_filename'] = os.path.relpath(rir_file_path, start=self.dataset_base_folder)
        else:
            rir_data = np.memmap(os.path.join(self.dataset_base_folder, relative_path), self.dtype, "r").reshape((1, -1))
            rir_info['rir_filename'] = kwargs['rir_filename']

        return rir_data, rir_info

    def _generate_rir_file_list(self):
        self.rir_file_list = []
        file_patterns = ["near/data/rsp*/*", "micarray/*/*/*/*/*/imp*.*"]
        for pattern in file_patterns:
            self.rir_file_list.extend(glob(os.path.join(self.dataset_base_folder, pattern)))
        
class ALL_dataset:
    def __init__(self, base_folder: str):
        self.base_folder = base_folder
        self.datasets = []
        for item in available_dataset:
            kwargs = {"base_folder": self.base_folder}
            self.datasets.append(Engine.create(item, **kwargs))

        self.lengths = []
        for dataset in self.datasets:
            self.lengths.append(len(dataset))

    def get_rir_data(self):
        # Only proive to get an rir_data randomly.
        weights = self.lengths / np.sum(self.lengths)
        dataset_index = np.random.choice(np.arange(len(self.datasets)), size=1, p=weights)[0]

        rir_data, rir_info = self.datasets[dataset_index].get_rir_data()
        if(len(rir_data.shape))>1:
            rir_data = rir_data[0]
        return rir_data, rir_info

    def __len__(self):
        return np.sum((self.lengths))

__all__ = ["Engine", "Engines"]
if __name__ == "__main__":

    pwd = os.curdir
    all = ALL_dataset(pwd)
    while True:
        rir_data, rir_info = all.get_rir_data()
        print(rir_data)

