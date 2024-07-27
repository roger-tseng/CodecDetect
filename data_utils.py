import os
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
import torchaudio
from torchaudio.functional import resample
from torch.utils.data import Dataset

from datasets import load_dataset

___author__ = "Haibin Wu"
__email__ = "f07921092@ntu.edu.tw"

DEV_SPK = ["p226", "p229"]
EVAL_SPK = ["p227", "p228"]


def genSpoof_list(dir_meta, codec_name="encodec", is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    others = []

    loaded_data = np.genfromtxt(dir_meta, delimiter=",", dtype=str)
    base_dir = None
    
    is_correct_codec = np.isin(loaded_data[:, 3], [codec_name, "genuine"])
    # selected_data = loaded_data[(loaded_data[:, 3] == codec_name ) | (loaded_data[loaded_data[:, 3] == "genuine"])]
    if is_train:
        is_correct_speaker = ~np.isin(loaded_data[:, 2], DEV_SPK + EVAL_SPK)
    elif is_eval:
        is_correct_speaker = np.isin(loaded_data[:, 2], EVAL_SPK)
    else:
        is_correct_speaker = np.isin(loaded_data[:, 2], DEV_SPK)

    selected_indices = np.where(is_correct_codec & is_correct_speaker)[0].tolist()
    selected_data = loaded_data[is_correct_codec & is_correct_speaker]

    if is_train:
        selected_data = selected_data[~np.isin(selected_data[:, 2], DEV_SPK + EVAL_SPK)]
    elif is_eval:
        selected_data = selected_data[np.isin(selected_data[:, 2], EVAL_SPK)]
    else:
        selected_data = selected_data[np.isin(selected_data[:, 2], DEV_SPK)]

    # for line in selected_data:
    #     wavpath, label, spkID, CodecName = line
    #     file_list.append(wavpath)
    return d_meta, selected_indices, base_dir, others


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
    def __init__(self, list_IDs, labels, base_dir, others=None, split="train"):
        self.list_IDs = list_IDs
        self.data = load_dataset("rogertseng/CodecFake", split=split)
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.resamplers = {
            24000: torchaudio.transforms.Resample(24000, 16000),
            44100: torchaudio.transforms.Resample(44100, 16000),
        }

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        data = self.data[key]

        utt_id = data["audio"]["path"]
        X = data["audio"]["array"]
        X = torch.from_numpy(X).float()
        sr = data["audio"]["sampling_rate"]
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
        y = 1 if data["label"] == "genuine" else 0
        others = (data["speaker_id"], data["codec_name"])
        return x_inp, y, utt_id, others
