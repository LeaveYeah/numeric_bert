BASEDIR="."
DATA_DIR="$BASEDIR/data/pubmedqa/bqal_new"
TASK_NAME="pubmedqaplus"
OUTPUT_DIR="$BASEDIR/out/pqaa_new/bqal_new"
# BIOBERT_DIR="$BASEDIR/pretrained_weights/biobert_large"
BIOBERT_DIR="$BASEDIR/out/pqaa_new"

python ./run_pubmedqa.py --model_name_or_path $BIOBERT_DIR --task_name=$TASK_NAME --do_train --do_predict  --max_seq_length=512 --data_dir $DATA_DIR --per_gpu_eval_batch_size=8 --per_gpu_train_batch_size=8 --learning_rate=2e-5 --num_train_epochs=4.0 --do_lower_case=false --output_dir=$OUTPUT_DIR --overwrite_output_dir --save_steps 25000 --gradient_accumulation_steps 2

#  --vocab_file=$BIOBERT_DIR/vocab_cased_pubmed_pmc_30k.txt --bert_config_file=$BIOBERT_DIR/bert_config_bio_58k_large.json 