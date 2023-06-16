### PREPROCESSING ###

python main.py --job=preprocessing --task_dataset=flickr8k
python main.py --job=preprocessing --task_dataset=flickr30k
python main.py --job=preprocessing --task_dataset=coco2014

### ANNOTATION ###
# NOTE: HRQ_VAE needs manual annotation process - see /task/annotating/hrqvae_annotating.ipynb
# NOTE: gpt_annotating requires openai api key

python main.py --task=annotating --job=onlyone_annotating --task_dataset=flickr8k
python main.py --task=annotating --job=synonym_annotating --task_dataset=flickr8k
python main.py --task=annotating --job=eda_annotating --task_dataset=flickr8k
python main.py --task=annotating --job=backtrans_annotating --task_dataset=flickr8k
python main.py --task=annotating --job=gpt_annotating --task_dataset=flickr8k
python main.py --task=annotating --job=budget_annotating --task_dataset=flickr8k

python main.py --task=annotating --job=onlyone_annotating --task_dataset=flickr30k
python main.py --task=annotating --job=synonym_annotating --task_dataset=flickr30k
python main.py --task=annotating --job=eda_annotating --task_dataset=flickr30k
python main.py --task=annotating --job=backtrans_annotating --task_dataset=flickr30k
python main.py --task=annotating --job=gpt_annotating --task_dataset=flickr30k
python main.py --task=annotating --job=budget_annotating --task_dataset=flickr30k

python main.py --task=annotating --job=onlyone_annotating --task_dataset=coco2014
python main.py --task=annotating --job=synonym_annotating --task_dataset=coco2014
python main.py --task=annotating --job=eda_annotating --task_dataset=coco2014
python main.py --task=annotating --job=backtrans_annotating --task_dataset=coco2014
python main.py --task=annotating --job=gpt_annotating --task_dataset=coco2014
python main.py --task=annotating --job=budget_annotating --task_dataset=coco2014

### EXPERIMENT ###

python main.py --job=training --task_dataset=flickr8k --annotation_mode=onlyone_en
python main.py --job=training --task_dataset=flickr8k --annotation_mode=synonym_en
python main.py --job=training --task_dataset=flickr8k --annotation_mode=eda_en
python main.py --job=training --task_dataset=flickr8k --annotation_mode=backtrans_en
python main.py --job=training --task_dataset=flickr8k --annotation_mode=hrqvae_en
python main.py --job=training --task_dataset=flickr8k --annotation_mode=gpt_en

python main.py --job=testing --task_dataset=flickr8k --annotation_mode=onlyone_en --decoding_strategy=beam --test_batch_size=1
python main.py --job=testing --task_dataset=flickr8k --annotation_mode=synonym_en --decoding_strategy=beam --test_batch_size=1
python main.py --job=testing --task_dataset=flickr8k --annotation_mode=eda_en --decoding_strategy=beam --test_batch_size=1
python main.py --job=testing --task_dataset=flickr8k --annotation_mode=backtrans_en --decoding_strategy=beam --test_batch_size=1
python main.py --job=testing --task_dataset=flickr8k --annotation_mode=hrqvae_en --decoding_strategy=beam --test_batch_size=1
python main.py --job=testing --task_dataset=flickr8k --annotation_mode=gpt_en --decoding_strategy=beam --test_batch_size=1

python main.py --job=training --task_dataset=flickr30k --annotation_mode=onlyone_en
python main.py --job=training --task_dataset=flickr30k --annotation_mode=synonym_en
python main.py --job=training --task_dataset=flickr30k --annotation_mode=eda_en
python main.py --job=training --task_dataset=flickr30k --annotation_mode=backtrans_en
python main.py --job=training --task_dataset=flickr30k --annotation_mode=hrqvae_en
python main.py --job=training --task_dataset=flickr30k --annotation_mode=gpt_en

python main.py --job=testing --task_dataset=flickr30k --annotation_mode=onlyone_en --decoding_strategy=beam --test_batch_size=1
python main.py --job=testing --task_dataset=flickr30k --annotation_mode=synonym_en --decoding_strategy=beam --test_batch_size=1
python main.py --job=testing --task_dataset=flickr30k --annotation_mode=eda_en --decoding_strategy=beam --test_batch_size=1
python main.py --job=testing --task_dataset=flickr30k --annotation_mode=backtrans_en --decoding_strategy=beam --test_batch_size=1
python main.py --job=testing --task_dataset=flickr30k --annotation_mode=hrqvae_en --decoding_strategy=beam --test_batch_size=1
python main.py --job=testing --task_dataset=flickr30k --annotation_mode=gpt_en --decoding_strategy=beam --test_batch_size=1

python main.py --job=training --task_dataset=coco2014 --annotation_mode=onlyone_en
python main.py --job=training --task_dataset=coco2014 --annotation_mode=synonym_en
python main.py --job=training --task_dataset=coco2014 --annotation_mode=eda_en
python main.py --job=training --task_dataset=coco2014 --annotation_mode=backtrans_en
python main.py --job=training --task_dataset=coco2014 --annotation_mode=hrqvae_en
python main.py --job=training --task_dataset=coco2014 --annotation_mode=gpt_en

python main.py --job=testing --task_dataset=coco2014 --annotation_mode=onlyone_en --decoding_strategy=beam --test_batch_size=1
python main.py --job=testing --task_dataset=coco2014 --annotation_mode=synonym_en --decoding_strategy=beam --test_batch_size=1
python main.py --job=testing --task_dataset=coco2014 --annotation_mode=eda_en --decoding_strategy=beam --test_batch_size=1
python main.py --job=testing --task_dataset=coco2014 --annotation_mode=backtrans_en --decoding_strategy=beam --test_batch_size=1
python main.py --job=testing --task_dataset=coco2014 --annotation_mode=hrqvae_en --decoding_strategy=beam --test_batch_size=1
python main.py --job=testing --task_dataset=coco2014 --annotation_mode=gpt_en --decoding_strategy=beam --test_batch_size=1
