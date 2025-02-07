model_name='gpt2-xl'
total_layer=48
target_acts="hidden_status"
layer_mode="layer-total"

for ((i = 0; i < total_layer; i++)); do
    echo "Task $model_name layer $i Started"
    python main.py --model_name $model_name --target_acts $target_acts --mode $layer_mode --layer_mode_layer $i
    echo "Task $model_name layer $i Ended"
done