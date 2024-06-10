#!/bin/bash
# evaluates all checkpoints listed in $models_list
# (assumes configs are two levels above the checkpoint)
# each ckpt is evaluated on all codecs listed in $codec_list
# resulting EERs are saved in eval/${train_codec_name}/${codec}/EER.txt

# source /home/b07901163/.bash_profile
# conda activate resyn
# cd /home/b07901163/CodecResynBenchmark

stage=1

models_list=(
    "ckpts/asvspoof_aasist/weights/AASIST.pth"
    "ckpts/asvspoof_aasist_l/weights/AASIST-L.pth"
    # "ckpts/SpeechTokenizer_AASIST-L_ep100_bs24/weights/epoch_4_0.544.pth"
    # "ckpts/academicodec_hifi_16k_320d_AASIST-L_ep5_bs24/weights/epoch_4_1.088.pth"
    # "ckpts/academicodec_hifi_16k_320d_large_uni_AASIST-L_ep5_bs24/weights/epoch_4_2.585.pth"
    # "exp_result/academicodec_hifi_24k_320d_AASIST-L_ep5_bs24/weights/epoch_2_4.082.pth"
    # "exp_result/audiodec_24k_320d_AASIST-L_ep5_bs24/weights/epoch_4_0.000.pth"
    # "ckpts/descript-audio-codec-16khz_AASIST-L_ep5_bs24/weights/epoch_4_0.000.pth"
    # "exp_result/descript-audio-codec-24khz_AASIST-L_ep5_bs24/weights/epoch_4_1.905.pth"
    # "exp_result/descript-audio-codec-44khz_AASIST-L_ep5_bs24/weights/epoch_4_0.136.pth"
    # "exp_result/encodec_24khz_AASIST-L_ep5_bs24/weights/epoch_4_0.000.pth"
    # "ckpts/funcodec-funcodec_en_libritts-16k-gr1nq32ds320_AASIST-L_ep5_bs24/weights/epoch_4_0.000.pth"
    # "ckpts/funcodec-funcodec_en_libritts-16k-gr8nq32ds320_AASIST-L_ep5_bs24/weights/epoch_4_0.000.pth"
    # "exp_result/funcodec-funcodec_en_libritts-16k-nq32ds320_AASIST-L_ep5_bs24/weights/epoch_4_0.000.pth"
    # "ckpts/funcodec-funcodec_en_libritts-16k-nq32ds640_AASIST-L_ep5_bs24/weights/epoch_4_0.000.pth"
    # "ckpts/funcodec-funcodec_zh_en_general_16k_nq32ds320_AASIST-L_ep5_bs24/weights/epoch_4_0.000.pth"
    # "ckpts/funcodec-funcodec_zh_en_general_16k_nq32ds640_AASIST-L_ep5_bs24/weights/epoch_2_0.000.pth"
)
for model_path in "${models_list[@]}"
do
    exp_path=$(dirname $(dirname ${model_path}))
    train_codec_name=$(basename ${exp_path})
    config=${exp_path}/config.conf
    # Put the model path here after training for evaluation
    # model_path="/media/hbwu/12TB/PublicData/aasist/exp_result/SpeechTokenizer_AASIST-L_ep100_bs24/weights/epoch_4_0.544.pth"

    # echo "Training with ${traincodecname} using ${config}"
    # if [ "$stage" -eq 0 ]; then
    #     python main.py --config ${config} \
    #                     --traincodecname ${traincodecname}
    # fi

    codec_list=(
        "valle"
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
        # "funcodec-funcodec_en_libritts-16k-nq32ds640"
        # "funcodec-funcodec_zh_en_general_16k_nq32ds320"
        # "funcodec-funcodec_zh_en_general_16k_nq32ds640"
    )
    if [ "$stage" -eq 1 ]; then
        for codec in "${codec_list[@]}"; do
            # echo "$(head -n 1 eval/${train_codec_name}/${codec}/EER.txt)%"
            # echo evaluate on ${codec}
            python main.py --eval \
                        --evalcodecname ${codec} \
                        --model_path ${model_path} \
                        --config ${config}
            for i in exp_result/${codec}_config_*
            do
                echo "saving to eval/${train_codec_name}/${codec}"
                mkdir -p eval/${train_codec_name}/${codec}
                mv ${i}/EER.txt eval/${train_codec_name}/${codec}/EER.txt
            done
        done
    fi
done
for model_path in "${models_list[@]}"
do
    exp_path=$(dirname $(dirname ${model_path}))
    train_codec_name=$(basename ${exp_path})

    codec_list=(
        "valle"
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
        # "funcodec-funcodec_en_libritts-16k-nq32ds640"
        # "funcodec-funcodec_zh_en_general_16k_nq32ds320"
        # "funcodec-funcodec_zh_en_general_16k_nq32ds640"
    )

    echo ${train_codec_name}
    for codec in "${codec_list[@]}"; do
        echo "$(head -n 1 eval/${train_codec_name}/${codec}/EER.txt)%"
    done
done