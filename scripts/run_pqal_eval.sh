BASEDIR="."
DATA_DIR="$BASEDIR/data/pubmedqa/pqal_folds"
TASK_NAME="pubmedqa"
OUTPUT_DIR="$BASEDIR/out/pqal"
# BIOBERT_DIR="$BASEDIR/pretrained_weights/biobert_large"
MODEL_DIR="$BASEDIR/out/pqal"
for i in {0..9}
do
    python -m torch.distributed.launch --nproc_per_node 2 ./examples/text-classification/run_glue.py --model_name_or_path $MODEL_DIR/pqal_fold$i --task_name=$TASK_NAME --do_predict  --max_seq_length=512 --data_dir $DATA_DIR/pqal_fold$i --per_gpu_eval_batch_size=4 --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=3.0 --do_lower_case=false --output_dir=$OUTPUT_DIR/pqal_fold$i --overwrite_output_dir --save_steps 1000
done