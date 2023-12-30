import os

# 指定包含.wav文件的文件夹的路径
folders = ["SpeechTokenizer", "descript-audio-codec-24khz", "funcodec-funcodec_en_libritts-16k-nq32ds640"]
base_path = "/media/hbwu/12TB/PublicData/resyn"
contents = os.listdir(base_path)
folders = [os.path.join(base_path, item) for item in contents if os.path.isdir(os.path.join(base_path, item))]

# 遍历每个文件夹
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    
    wav_file_count = 0

    # 检查文件夹是否存在
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # 遍历文件夹中的所有文件
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 检查文件扩展名是否为.wav
                if file.endswith(".wav"):
                    wav_file_count += 1

    # 打印.wav文件的数量
    print("{} 总共找到 {} 个.wav文件".format(folder, wav_file_count))
