BASEDIR="."
DATA_DIR="$BASEDIR/data/pubmedqa/pqaa_new"
TASK_NAME="pubmedqaplus"
OUTPUT_DIR="$BASEDIR/out/pqaa_new"
# BIOBERT_DIR="$BASEDIR/pretrained_weights/biobert_large"
BIOBERT_DIR="$BASEDIR/pretrained_weights/biobert_v1.1_pubmed"

python ./run_pubmedqa.py --model_name_or_path $BIOBERT_DIR --task_name=$TASK_NAME --do_train --do_eval --max_seq_length=512 --data_dir $DATA_DIR --per_gpu_eval_batch_size=8 --per_gpu_train_batch_size=8 --learning_rate=2e-5 --num_train_epochs=3.0 --do_lower_case=false --output_dir=$OUTPUT_DIR --overwrite_output_dir --save_steps 25000 --gradient_accumulation 2

#  --vocab_file=$BIOBERT_DIR/vocab_cased_pubmed_pmc_30k.txt --bert_config_file=$BIOBERT_DIR/bert_config_bio_58k_large.json 