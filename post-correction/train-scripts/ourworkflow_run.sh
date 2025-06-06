# Set the experiment number
EXPERIMENT=data_efficiency_38

# Loop over folds from 1 to 3
for FOLD in {1..1}
do
    echo "Processing fold $FOLD..."

    /home/ocr_proj/.conda/envs/ByT5/bin/python /home/ocr_proj/OCR/post_correction/pe-ocr-sanskrit/train-scripts/finetune_ByT5-sanskrit.py --experiment $EXPERIMENT --fold $FOLD
    /home/ocr_proj/.conda/envs/ByT5/bin/python /home/ocr_proj/OCR/post_correction/pe-ocr-sanskrit/train-scripts/inference-byt5-sanskrit.py --experiment $EXPERIMENT --fold $FOLD

    # getting the error counts..
    TRAIN_PATH=/home/ocr_proj/OCR/post_correction/pe-ocr-sanskrit/data/experiment_$EXPERIMENT/train_fold_$FOLD.csv
    TEST_PATH=/home/ocr_proj/OCR/post_correction/pe-ocr-sanskrit/outputs/experiment_$EXPERIMENT/test_fold_$FOLD/analysis.csv
    /home/ocr_proj/.conda/envs/ByT5/bin/python /home/ocr_proj/OCR/post_correction/pe-ocr-sanskrit/train-scripts/denoise.py --csv_file $TRAIN_PATH --train 
    /home/ocr_proj/.conda/envs/ByT5/bin/python /home/ocr_proj/OCR/post_correction/pe-ocr-sanskrit/train-scripts/denoise.py --csv_file $TEST_PATH

    echo "Completed fold $FOLD"
done




