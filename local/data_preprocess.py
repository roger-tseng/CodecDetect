import os
import shutil
import numpy as np
from tqdm import tqdm
import soundfile as sf

# codec files source path
path = "/media/hbwu/12TB/PublicData/resyn"
contents = os.listdir(path)
source_folders = [os.path.join(path, item) for item in contents if os.path.isdir(os.path.join(path, item))]

target_folder = "/media/hbwu/12TB/PublicData/CodecData/processed"  # target path


data_list = []
data_list.append(["wavpath", "label", "spkID", "CodecName"])
data_list.append([target_folder, "dummy", "dummy", "dummy"])
csv_file = "/media/hbwu/12TB/PublicData/CodecData/datalist.csv"  # datalist path

for source_folder in source_folders:
    wav_files = [f for f in os.listdir(source_folder) if f.endswith(".wav")]

    for wav_file in tqdm(wav_files, desc=source_folder):
        spk, wav_name = wav_file.split("_")

        codec_name = source_folder.split("/")[-1]
        new_filename = f"{codec_name}+{wav_file}"

        source_path = os.path.join(source_folder, wav_file)
        target_path = os.path.join(target_folder, new_filename)
        try:
            X, _ = sf.read(source_path)
            shutil.move(source_path, target_path)
            label = "spoofing" if source_folder.split("/")[-1] != "genuine" else "genuine"
            data_info = [new_filename, label, spk, codec_name]
            data_list.append(data_info)
        except:
            print(f"{source_path} is damaged")

np.savetxt(csv_file, data_list, delimiter=",", fmt="%s")

## Load data
# loaded_data = np.genfromtxt(csv_file, delimiter=',', dtype=str)
# selected_data = loaded_data[(loaded_data[2:, 2] == 'spkID1') & (loaded_data[2:, 3] == 'CodecName3')]
