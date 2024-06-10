#!/bin/bash

# source /home/b07901163/.bash_profile
# conda activate resyn
# cd /home/b07901163/CodecResynBenchmark

stage=1

exp_path="exp_result/audiodec_24k_320d_AASIST-L_ep5_bs24"
config=${exp_path}/config.conf
model_path=${exp_path}/weights/epoch_4_0.000.pth
# Put the model path here after training for evaluation
# model_path="/media/hbwu/12TB/PublicData/aasist/exp_result/SpeechTokenizer_AASIST-L_ep100_bs24/weights/epoch_4_0.544.pth"

# echo "Training with ${traincodecname} using ${config}"
# if [ "$stage" -eq 0 ]; then
#     python main.py --config ${config} \
#                     --traincodecname ${traincodecname}
# fi

codec_list=(
    # "SpeechTokenizer"
    # "academicodec_hifi_16k_320d"
    # "academicodec_hifi_16k_320d_large_uni"
    # "academicodec_hifi_24k_320d"
    # "audiodec_24k_320d"
    # "descript-audio-codec-16khz"
    # "descript-audio-codec-24khz"
    # "descript-audio-codec-44khz"
    # "encodec_24khz"
    # "funcodec-funcodec_en_libritts-16k-gr1nq32ds320"
    # "funcodec-funcodec_en_libritts-16k-gr8nq32ds320"
    # "funcodec-funcodec_en_libritts-16k-nq32ds320"
    "funcodec-funcodec_en_libritts-16k-nq32ds640"
    # "funcodec-funcodec_zh_en_general_16k_nq32ds320"
    # "funcodec-funcodec_zh_en_general_16k_nq32ds640"
)
if [ "$stage" -eq 1 ]; then
    for codec in "${codec_list[@]}"; do
        echo evaluate on ${codec}
        python main.py --eval \
                    --evalcodecname ${codec} \
                    --model_path ${model_path} \
                    --config ${config}
    done
    for i in exp_result/${codec}_config_*
    do
        if [ ! -d "${i}/metrics" ]; then
            echo "remove ${i}"
            #rm $i -rf
        fi
        head -n 1 ${i}/EER.txt
    done
fi
