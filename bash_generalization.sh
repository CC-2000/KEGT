train_dataset_number_list=(1 2 4 8 16 32 64 128)

model_name="llama3.1-8b"
target_acts="hidden_status"

method="main"

for train_dataset_num in "${train_dataset_number_list[@]}"; do

    if [ "$method" = "ccs" ]; then
        echo "ccs"
        echo "Task number of $train_dataset_num training datasets Started"
        python ccs_main.py --model_name $model_name --target_acts $target_acts --mode generalization --train_dataset_number $train_dataset_num
        echo "Task number of $train_dataset_num training datasets Ended"
    elif [ "$method" = "main" ]; then
        echo "ours"
        echo "Task number of $train_dataset_num training datasets Started"
        python main.py --model_name $model_name --target_acts $target_acts --mode generalization --train_dataset_number $train_dataset_num
        echo "Task number of $train_dataset_num training datasets Ended"
    else
        echo "NONONO"
    fi
    

done

echo $model_name