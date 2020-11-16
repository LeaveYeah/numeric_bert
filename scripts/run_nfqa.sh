BASEDIR=".."
DATA_DIR="$BASEDIR/data/BioNumFactQA/nfqa_fold"
TASK_NAME="squad"
OUTPUT_DIR="$BASEDIR/out/nfqa1/nfqa_fold"
# BIOBERT_DIR="$BASEDIR/pretrained_weights/biobert_large"
# MODEL_DIR="$BASEDIR/out/bioasq"
MODEL_DIR="$BASEDIR/pretrained_weights/biobert_v1.1_pubmed"

for i in {0..9}
do
    python ../examples/question-answering/run_squad.py --model_type bert --model_name_or_path $MODEL_DIR --do_train --do_eval  --max_seq_length=512 --data_dir "$DATA_DIR$i" --train_file train-v1.1.json --predict_file dev-v1.1.json  --per_gpu_eval_batch_size=8 --do_lower_case --per_gpu_train_batch_size=8 --learning_rate=2e-5 --num_train_epochs=4.0 --output_dir=$OUTPUT_DIR$i --overwrite_output_dir --save_steps 1000  --gradient_accumulation_steps 2
    #  --vocab_file=$BIOBERT_DIR/vocab_cased_pubmed_pmc_30k.txt --bert_config_file=$BIOBERT_DIR/bert_config_bio_58k_large.json 
done