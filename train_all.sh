for codec in "academicodec_hifi_24k_320d" "descript-audio-codec-24khz" "descript-audio-codec-44khz" "encodec_24khz" #"audiodec_24k_320d"
do
    bash train.sh $codec ./config/AASIST-L.conf
done