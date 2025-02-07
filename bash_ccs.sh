layer_list=(3 27 39)

model_name="llama-2-13b-hf"
target_acts="hidden_status"

for cur_layer in "${layer_list[@]}"; do
    echo "Task of $model_name $cur_layer training datasets Started"
    python ccs_main.py --model_name $model_name --target_acts $target_acts --layer $cur_layer
    echo "Task of $model_name $cur_layer training datasets Ended"
done
