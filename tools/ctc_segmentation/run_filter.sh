#!/bin/bash

MODEL_NAME_OR_PATH=""
INPUT_AUDIO_DIR=""
MANIFEST=""
BATCH_SIZE=4

# Thresholds for filtering
CER_THRESHOLD=100
WER_THRESHOLD=100
CER_START_THRESHOLD=100
CER_END_THRESHOLD=100


for ARG in "$@"
do
    key=$(echo $ARG | cut -f1 -d=)
    value=$(echo $ARG | cut -f2 -d=)

    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare $v="${value}"
    fi
done

if [[ -z $MODEL_NAME_OR_PATH ]] || [[ -z $INPUT_AUDIO_DIR ]] || [[ -z $MANIFEST ]]; then
  echo "Usage: $(basename "$0")
  --MODEL_NAME_OR_PATH=[path to .nemo ASR model or a pre-trained model name to use for metrics calculation]
  --INPUT_AUDIO_DIR=[path to original directory with audio files used for segmentation]
  --MANIFEST=[path to manifest file generated during segmentation]"
  exit 1
fi

echo "--- Adding transcripts to ${MANIFEST} using ${MODEL_NAME_OR_PATH} ---"
if [[ ${MODEL_NAME_OR_PATH,,} == *".nemo" ]]; then
  ARG_MODEL="model_path";
else
  ARG_MODEL="pretrained_name";
fi

# Add transcripts to the manifest file, ASR model predictions will be stored under "pred_text" field
python /home/ebakhturina/NeMo/examples/asr/transcribe_speech.py \
$ARG_MODEL=$MODEL_NAME_OR_PATH \
dataset_manifest=$MANIFEST \
output_filename=${MANIFEST}_transcribed.json || exit

echo "--- Calculation metrics ---"
python /home/ebakhturina/misc_scripts/ctc_segmentation/benchmark/calc_metrics.py \
--input=${MANIFEST}_transcribed.json \
--output=${MANIFEST}_transcribed_metrics.json \
--asr_pred=pred_text

rm -rf ${MANIFEST}_transcribed.json

echo "--- Filter out samples based on thresholds ---"
echo "CER_THRESHOLD = ${CER_THRESHOLD}"
echo "WER_THRESHOLD = ${WER_THRESHOLD}"
echo "CER_START_THRESHOLD = ${CER_START_THRESHOLD}"
echo "CER_END_THRESHOLD = ${CER_END_THRESHOLD}"

python /home/ebakhturina/misc_scripts/ctc_segmentation/benchmark/agg_metrics.py \
--manifest=${MANIFEST}_transcribed_metrics.json \
--audio_dir=/home/ebakhturina/data/segmentation/benchmark/DEL/audio/

# clean up
##rm -rf ${OUTPUT_DIR}/processed

#python /home/ebakhturina/NeMo/tools/speech_data_explorer/data_explorer.py \
#--disable-caching-metrics \
#${OUTPUT_DIR}/manifests/all_transcribed.json