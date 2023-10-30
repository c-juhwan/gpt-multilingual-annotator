### PREPROCESSING ###
DEVICE=0
DATASET=multi30k

CUDA_VISIBLE_DEVICES=${DEVICE} python main.py --task=machine_translation --job=preprocessing --task_dataset=${DATASET}

### ANNOTATION ###
CUDA_VISIBLE_DEVICES=${DEVICE} python main.py --task=annotating_mt --job=translation_annotating --task_dataset=${DATASET} --annotation_mode=translated_de
python main.py --task=annotating_mt --job=gpt_annotating --task_dataset=${DATASET} --gpt_model_version=gpt-4 --annotation_mode=original_de

### EXPERIMENT ###

CUDA_VISIBLE_DEVICES=${DEVICE} python main.py --task=machine_translation --job=training --optimize_objective=loss --task_dataset=multi30k --annotation_mode=original_de
CUDA_VISIBLE_DEVICES=${DEVICE} python main.py --task=machine_translation --job=testing --optimize_objective=loss --task_dataset=multi30k --annotation_mode=original_de
CUDA_VISIBLE_DEVICES=${DEVICE} python main.py --task=machine_translation --job=training --optimize_objective=loss --task_dataset=multi30k --annotation_mode=translated_de
CUDA_VISIBLE_DEVICES=${DEVICE} python main.py --task=machine_translation --job=testing --optimize_objective=loss --task_dataset=multi30k --annotation_mode=translated_de
CUDA_VISIBLE_DEVICES=${DEVICE} python main.py --task=machine_translation --job=training --optimize_objective=loss --task_dataset=multi30k --annotation_mode=gpt_de --gpt_model_version=gpt-4
CUDA_VISIBLE_DEVICES=${DEVICE} python main.py --task=machine_translationf --job=testing --optimize_objective=loss --task_dataset=multi30k --annotation_mode=gpt_de --gpt_model_version=gpt-4
