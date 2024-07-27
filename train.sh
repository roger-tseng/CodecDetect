#!/bin/bash

stage=0
traincodecname=$1 # SpeechTokenizer
config=${2:-./config/AASIST-L.conf}

if [ ! -e datalist.csv ]; then
    echo "Downloading datalist.csv"
    wget https://huggingface.co/datasets/rogertseng/CodecFake/resolve/main/datalist.csv?download=true
fi

echo "Training with ${traincodecname} using ${config}"
python main.py \
    --config ${config} \
    --traincodecname ${traincodecname} \
    --database_path datalist.csv

codec_list=(
    "SpeechTokenizer"
    "academicodec_hifi_16k_320d"
    "academicodec_hifi_16k_320d_large_uni"
    "academicodec_hifi_24k_320d"
    "audiodec_24k_320d"
    "descript-audio-codec-16khz"
    "descript-audio-codec-24khz"
    "descript-audio-codec-44khz"
    "encodec_24khz"
    "funcodec-funcodec_en_libritts-16k-gr1nq32ds320"
    "funcodec-funcodec_en_libritts-16k-gr8nq32ds320"
    "funcodec-funcodec_en_libritts-16k-nq32ds320"
    "funcodec-funcodec_en_libritts-16k-nq32ds640"
    "funcodec-funcodec_zh_en_general_16k_nq32ds320"
    "funcodec-funcodec_zh_en_general_16k_nq32ds640"
)
