#!/bin/bash
# evaluates all checkpoints listed in $models_list
# (assumes configs are two levels above the checkpoint)
# each ckpt is evaluated on all codecs listed in $codec_list
# resulting EERs are saved in eval/${train_codec_name}/${codec}/EER.txt

stage=1

models_list=(
    "ckpts/asvspoof_aasist/weights/AASIST.pth"
    "ckpts/asvspoof_aasist_l/weights/AASIST-L.pth"
    # add your other trained checkpoints hrere
)
for model_path in "${models_list[@]}"
do
    exp_path=$(dirname $(dirname ${model_path}))
    train_codec_name=$(basename ${exp_path})
    config=${exp_path}/config.conf

    codec_list=(
        "valle"
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
    if [ "$stage" -eq 1 ]; then
        for codec in "${codec_list[@]}"; do
            python main.py --eval \
                        --evalcodecname ${codec} \
                        --model_path ${model_path} \
                        --config ${config}
            # Aggregate EER results
            for i in exp_result/${codec}_config_*/eval
            do
                echo "saving to eval/${train_codec_name}/${codec}"
                mkdir -p eval/${train_codec_name}/${codec}
                cp ${i}/EER.txt eval/${train_codec_name}/${codec}/EER.txt
            done
        done
    fi
done