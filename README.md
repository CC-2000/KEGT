# KEGT

This is the official implementation of KEGT: Token Knowledge——A New Perspective For Knowledge in Large Language Models.

The relevant data can be downloaded from [here](https://drive.google.com/drive/folders/16fckNf0_BaEo2OMLzJkWvNbeQfLNvVfo?usp=drive_link).

## For ours cleaned KASS dataset

### Generate a dataset for a specific model

python dataset_make.py --model_name llama3.1-8b --data_root "./data/KASS" --dataset_name KASS --other_datasets None --device_id 1

### Generate activations for all layers of a specific model

python generate_acts.py --model_name llama3.1-8b --device_id 1 --output_dir hidden_status

### Training probes with hidden status during inference

python main.py --model_name llama3.1-8b --target_acts hidden_status --mode main


## For other open-ended dataset

python dataset_make.py --model_name llama3.1-8b --device_id 1 --other_datasets WebQ

python generate_acts.py --model_name llama3.1-8b --layers 14 --output_dir hidden_status --datasets WebQ --device_id 1 --data_root "./data/WebQ"

python main.py --config_file other_dataset_exp.yaml --model_name llama3.1-8b

## For low training data setting

bash bash_generalization.sh