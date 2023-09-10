conda activate OpenChatKitVB

accelerate launch scripts/summarization/run_summarization_no_trainer.py \
    --model_name_or_path ./build/model \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ./build/accelerate