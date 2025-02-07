model_name="Qwen2.5-7B"

echo "Task $model_name zero-shot Started"
python X-shot_main.py --model_name $model_name --config_file zero-shot_exp.yaml

echo "Task $model_name few-shot Started"
python X-shot_main.py --model_name $model_name --config_file few-shot_exp.yaml

echo "Task $model_name zero-shot Ended"
echo "task $model_name few-shot Ended"