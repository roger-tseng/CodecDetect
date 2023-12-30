#!/bin/bash

stage=0
config="./config/AASIST-L.conf"
traincodecname="SpeechTokenizer"

if [ "$stage" -eq 0 ]; then
    echo training with config ${config}
    python main.py --config ${config} \
                    --traincodecname ${traincodecname}
fi

codec_list=(
    "SpeechTokenizer"
    "descript-audio-codec-24khz"
    "funcodec-funcodec_en_libritts-16k-nq32ds640"
    "academicodec_hifi_16k_320d"
    "descript-audio-codec-44khz"
    "funcodec-funcodec_zh_en_general_16k_nq32ds320"
    "academicodec_hifi_16k_320d_large_uni"
    "encodec_24khz"
    "funcodec-funcodec_zh_en_general_16k_nq32ds640"
    "academicodec_hifi_24k_320d"
    "funcodec-funcodec_en_libritts-16k-gr1nq32ds320"
    "audiodec_24k_320d"
    "funcodec-funcodec_en_libritts-16k-gr8nq32ds320"
    "descript-audio-codec-16khz"
    "funcodec-funcodec_en_libritts-16k-nq32ds320"
)
if [ "$stage" -eq 1 ]; then
    for codec in "${my_list[@]}"; do
        echo evaluate on ${codec}
        python main.py --eval \
                    --evalcodecname ${codec} \
                    --config ${config}
    done
fi
