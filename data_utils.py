import os
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
import torchaudio
from torchaudio.functional import resample
from torch.utils.data import Dataset

___author__ = "Haibin Wu"
__email__ = "f07921092@ntu.edu.tw"

DEV_SPK = ["p226", "p229"]
EVAL_SPK = ["p227", "p228"]


def genSpoof_list(dir_meta, codec_name="encodec", is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    others = []

    loaded_data = np.genfromtxt(dir_meta, delimiter=",", dtype=str)
    base_dir = str(loaded_data[1, 0])
    selected_data = loaded_data[np.isin(loaded_data[:, 3], [codec_name, "genuine"])]
    # selected_data = loaded_data[(loaded_data[:, 3] == codec_name ) | (loaded_data[loaded_data[:, 3] == "genuine"])]

    if is_train:
        selected_data = selected_data[~np.isin(selected_data[:, 2], DEV_SPK + EVAL_SPK)]
    elif is_eval:
        selected_data = selected_data[np.isin(selected_data[:, 2], EVAL_SPK)]
    else:
        selected_data = selected_data[np.isin(selected_data[:, 2], DEV_SPK)]

    for line in selected_data:
        wavpath, label, spkID, CodecName = line
        file_list.append(wavpath)
        others.append([spkID, CodecName])
        d_meta[wavpath] = 1 if label == "genuine" else 0
    return d_meta, file_list, base_dir, others


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: Tensor, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt : stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = torch.tile(x, (num_repeats,))[:max_len]
    return padded_x

class Dataset_Codec_Antispoofing(Dataset):
    def __init__(self, list_IDs, labels, base_dir, others=None):
        """self.list_IDs	: list of strings (each string: utt key),
        self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.others = others  # include [spk, codecname]
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.resamplers = {
            24000: torchaudio.transforms.Resample(24000, 16000),
            44100: torchaudio.transforms.Resample(44100, 16000),
        }

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        others = self.others[index]
        X, sr = sf.read(os.path.join(self.base_dir, key))
        X = torch.from_numpy(X).float()
        # assert sr == 16000
        
        if sr != 16000:
            if sr in self.resamplers:
                X = self.resamplers[sr](X)
            else:
                X = resample(
                    X, sr, 16000
                )
        
        assert len(X.shape) == 1
        x_inp = pad_random(X, self.cut)
        y = self.labels[key]
        return x_inp, y, key, others
