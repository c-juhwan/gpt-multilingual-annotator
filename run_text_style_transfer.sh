### PREPROCESSING ###
# NOTE: you need to download the dataset first (which requires approval)
# NOTE: you need to execute xformal_process.ipynb first in task/text_style_transfer/

python main.py --task=text_style_transfer --job=preprocessing --task_dataset=gyafc_en
python main.py --task=text_style_transfer --job=preprocessing --task_dataset=xformal_fr
python main.py --task=text_style_transfer --job=preprocessing --task_dataset=xformal_pt
python main.py --task=text_style_transfer --job=preprocessing --task_dataset=xformal_it

### ANNOTATION ###
DATASET_LIST=(xformal_fr xformal_pt xformal_it)

for DATASET in ${DATASET_LIST[@]}; do
    python main.py --task=annotating_tst --job=translation_annotating --task_dataset=${DATASET}
    python main.py --task=annotating_tst --job=gpt_annotating --task_dataset=${DATASET} --gpt_model_version=gpt-4
done

### EXPERIMENT ###

# Train Style Classifier First
python main.py --task=style_classification --job=training --task_dataset=xformal_fr --annotation_mode=original_fr
python main.py --task=style_classification --job=testing --task_dataset=xformal_fr --annotation_mode=original_fr
python main.py --task=style_classification --job=training --task_dataset=xformal_pt --annotation_mode=original_pt
python main.py --task=style_classification --job=testing --task_dataset=xformal_pt --annotation_mode=original_pt
python main.py --task=style_classification --job=training --task_dataset=xformal_it --annotation_mode=original_it
python main.py --task=style_classification --job=testing --task_dataset=xformal_it --annotation_mode=original_it

# Experiment 4.4 - Training & Testing & Style Classification Inference
python main.py --task=text_style_transfer --job=training --optimize_objective=loss --task_dataset=xformal_fr --annotation_mode=translated_fr
python main.py --task=text_style_transfer --job=testing --optimize_objective=loss --task_dataset=xformal_fr --annotation_mode=translated_fr
python main.py --task=style_classification --job=inference --task_dataset=xformal_fr --annotation_mode=translated_fr
python main.py --task=text_style_transfer --job=training --optimize_objective=loss --task_dataset=xformal_fr --annotation_mode=gpt_fr --gpt_model_version=gpt-4
python main.py --task=text_style_transfer --job=testing --optimize_objective=loss --task_dataset=xformal_fr --annotation_mode=gpt_fr --gpt_model_version=gpt-4
python main.py --task=style_classification --job=inference --task_dataset=xformal_fr --annotation_mode=gpt_fr --gpt_model_version=gpt-4

python main.py --task=text_style_transfer --job=training --optimize_objective=loss --task_dataset=xformal_pt --annotation_mode=translated_pt
python main.py --task=text_style_transfer --job=testing --optimize_objective=loss --task_dataset=xformal_pt --annotation_mode=translated_pt
python main.py --task=style_classification --job=inference --task_dataset=xformal_pt --annotation_mode=translated_pt
python main.py --task=text_style_transfer --job=training --optimize_objective=loss --task_dataset=xformal_pt --annotation_mode=gpt_pt --gpt_model_version=gpt-4
python main.py --task=text_style_transfer --job=testing --optimize_objective=loss --task_dataset=xformal_pt --annotation_mode=gpt_pt --gpt_model_version=gpt-4
python main.py --task=style_classification --job=inference --task_dataset=xformal_pt --annotation_mode=gpt_pt --gpt_model_version=gpt-4

python main.py --task=text_style_transfer --job=training --optimize_objective=loss --task_dataset=xformal_it --annotation_mode=translated_it
python main.py --task=text_style_transfer --job=testing --optimize_objective=loss --task_dataset=xformal_it --annotation_mode=translated_it
python main.py --task=style_classification --job=inference --task_dataset=xformal_it --annotation_mode=translated_it
python main.py --task=text_style_transfer --job=training --optimize_objective=loss --task_dataset=xformal_it --annotation_mode=gpt_it --gpt_model_version=gpt-4
python main.py --task=text_style_transfer --job=testing --optimize_objective=loss --task_dataset=xformal_it --annotation_mode=gpt_it --gpt_model_version=gpt-4
python main.py --task=style_classification --job=inference --task_dataset=xformal_it --annotation_mode=gpt_it --gpt_model_version=gpt-4
