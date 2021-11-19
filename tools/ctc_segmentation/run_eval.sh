MODEL_NAME_OR_PATH="stt_en_conformer_ctc_large"
# STEP #6
# Add transcripts to the manifest file, ASR model predictions will be stored under "pred_text" field
if [[ ${MODEL_NAME_OR_PATH,,} == *".nemo" ]]; then
  ARG_MODEL="model_path";
else
  ARG_MODEL="pretrained_name";
fi

BASE_DIR="/home/ebakhturina/data/segmentation/benchmark"
#"QuartzNet15x5Base-Ensubset_fix"
"QuartzNet15x5Base-Ensubset_nofix"
for OUTPUT_DIR in "stt_en_citrinet_512_gamma_0_25subset" "stt_en_citrinet_512_gamma_0_25subset_nofix"
do
  OUTPUT_DIR=${BASE_DIR}/${OUTPUT_DIR}
  python /home/ebakhturina/NeMo/examples/asr/transcribe_speech.py \
  $ARG_MODEL=$MODEL_NAME_OR_PATH \
  dataset_manifest=${OUTPUT_DIR}/manifests/manifest.json \
  output_filename=${OUTPUT_DIR}/manifests/all_transcribed_conformer.json \
  batch_size=16 || exit

  # calculate metrics
  python /home/ebakhturina/misc_scripts/ctc_segmentation/benchmark/calc_metrics.py \
  --input=${OUTPUT_DIR}/manifests/all_transcribed_conformer.json \
  --output=${OUTPUT_DIR}/manifests/all_transcribed_conformer_metrics.json \
  --asr_pred=pred_text

  # filter
  python /home/ebakhturina/misc_scripts/ctc_segmentation/benchmark/agg_metrics.py \
  --manifest=${OUTPUT_DIR}/manifests/all_transcribed_conformer_metrics.json \
  --audio_dir=/home/ebakhturina/data/segmentation/benchmark/DEL/audio/
done