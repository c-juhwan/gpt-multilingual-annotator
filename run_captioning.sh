clear
### PREPROCESSING ###
# NOTE: You need to download the dataset first
# NOTE: for (aide, uit_viic, new_lv), you need to run .ipynb files in /task/captioning/ first

python main.py --job=preprocessing --task_dataset=flickr8k
python main.py --job=preprocessing --task_dataset=flickr30k
python main.py --job=preprocessing --task_dataset=coco2014
python main.py --job=preprocessing --task_dataset=${DATASET}
python main.py --job=preprocessing --task_dataset=aide
python main.py --job=preprocessing --task_dataset=uit_viic
python main.py --job=preprocessing --task_dataset=new_lv

### ANNOTATION ###

DATASET_LIST1=(flickr8k flickr30k coco2014)
for DATASET in ${DATASET_LIST1[@]}; do
    python main.py --task=annotating --job=gpt_annotating --task_dataset=${DATASET} --gpt_model_version=gpt-3.5-turbo
    python main.py --task=annotating --job=budget_annotating --task_dataset=${DATASET}
done
# GPT-4 annotation is COCO2014 only
python main.py --task=annotating --job=gpt_annotating --task_dataset=coco2014 --gpt_model_version=gpt-4

DATASET_LIST2=(aide uit_viic new_lv)
for DATASET in ${DATASET_LIST2[@]}; do
    python main.py --task=annotating --job=translation_annotating --task_dataset=${DATASET}
    python main.py --task=annotating --job=gpt_annotating --task_dataset=${DATASET} --gpt_model_version=gpt-4
done

### EXPERIMENT ###

# Experiment 4.2 - Cost Efficiency
for DATASET in ${DATASET_LIST1[@]}; do
    python main.py --job=training --task_dataset=${DATASET} --annotation_mode=budget_en
    python main.py --job=training --task_dataset=${DATASET} --annotation_mode=gpt_en --gpt_model_version=gpt-3.5-turbo
    python main.py --job=testing --task_dataset=${DATASET} --annotation_mode=budget_en --decoding_strategy=beam --test_batch_size=1
    python main.py --job=testing --task_dataset=${DATASET} --annotation_mode=gpt_en --gpt_model_version=gpt-3.5-turbo --decoding_strategy=beam --test_batch_size=1
done

# Experiment 4.3.1 - Korean
python main.py --job=training --task_dataset=coco2014 --annotation_mode=aihub_ko
python main.py --job=training --task_dataset=coco2014 --annotation_mode=gpt_ko --gpt_model_version=gpt-4
python main.py --job=testing --task_dataset=coco2014 --annotation_mode=aihub_ko --decoding_strategy=beam --test_batch_size=1
python main.py --job=testing --task_dataset=coco2014 --annotation_mode=gpt_ko --gpt_model_version=gpt-4 --decoding_strategy=beam --test_batch_size=1

# Experiment 4.3.2 - Vietnamese
python main.py --job=training --task_dataset=uit_viic --annotation_mode=original_vie
python main.py --job=testing --task_dataset=uit_viic --annotation_mode=original_vie --decoding_strategy=beam --test_batch_size=1

python main.py --job=training --task_dataset=uit_viic --annotation_mode=translated_vie
python main.py --job=testing --task_dataset=uit_viic --annotation_mode=translated_vie --decoding_strategy=beam --test_batch_size=1

python main.py --job=training --task_dataset=uit_viic --annotation_mode=gpt_vie  --gpt_model_version=gpt-4
python main.py --job=testing --task_dataset=uit_viic --annotation_mode=gpt_vie  --gpt_model_version=gpt-4 --decoding_strategy=beam --test_batch_size=1

# Experiment 4.3.3 - Polish
python main.py --job=training --task_dataset=aide --annotation_mode=original_pl
python main.py --job=testing --task_dataset=aide --annotation_mode=original_pl --decoding_strategy=beam --test_batch_size=1

python main.py --job=training --task_dataset=aide --annotation_mode=translated_pl
python main.py --job=testing --task_dataset=aide --annotation_mode=translated_pl --decoding_strategy=beam --test_batch_size=1

python main.py --job=training --task_dataset=aide --annotation_mode=gpt_pl --gpt_model_version=gpt-4
python main.py --job=testing --task_dataset=aide --annotation_mode=gpt_pl --gpt_model_version=gpt-4 --decoding_strategy=beam --test_batch_size=1

# Experiment 4.5 - Dataset Construction (Latvian)
python main.py --job=training --task_dataset=new_lv --annotation_mode=translated_lv
python main.py --job=testing --task_dataset=new_lv --annotation_mode=translated_lv --decoding_strategy=beam --test_batch_size=1

python main.py --job=training --task_dataset=new_lv --annotation_mode=gpt_lv --gpt_model_version=gpt-4
python main.py --job=testing --task_dataset=new_lv --annotation_mode=gpt_lv --gpt_model_version=gpt-4 --decoding_strategy=beam --test_batch_size=1