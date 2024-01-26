# [EACL 2024 Findings] GPTs are (Multilingual) Annotators for Sequence Generation Tasks

## Annotation Software - Requirements

- openai == 0.27.2

## Annotation Software - Usage

### Prepare the data

Please prepare a json file, following the format of `original/example_empty.json`. There is also a complete example in `original/example_vietnamese.json`. You have to modify the `prompt` field regarding your target language.

#### Configuration

- `task`: The task name. Currently, this annotator supports `image_captioning`, `text_style_transfer` and `machine_translation`.
- `prompt`: The prompt for the annotator. You have to modify this field regarding your target language.
- `source_lang`: Source language, i.e. `en`. Currently, this annotator only supports English as the source language.
- `target_lang`: Target language, e.g. `vie`.
- `gpt_model_version`: The version of GPT model. e.g. `gpt-4` or `gpt-3.5-turbo`.

#### Data Field

- `file_name`: The file name of the image.
- `split`: If you don't have explicit train/valid/test split, you can just set this field to `train` for all the data.
- `source_captions`: You must provide at least one source caption.
- `target_gold_captions`: You don't nessarily have to provide the target gold captions. If you don't have any gold captions, you can simply set this field empty, i.e. `[]`.
- `target_silver_captions`: This field should be set empty, i.e. `[]`, in your original data. This field will be filled by the annotator.

### Run the annotator script

```bash
python annotator.py --input=./original/example_vietnamese.json --output=./result/vie_annotated_example.json
python annotator.py --input=./original/example_tst_french.json --output=./result/fr_tst_annotated_example.json
python annotator.py --input=./original/example_mt_estonian_test.json --output=./result/et_mt_annotated_example.json
```

#### Arguments

- `input`: The input json file.
- `output`: The output json file.
- `random_selection`: If not 0, it will be used as a random seed value and the annotator will randomly select one of the source captions as the annotation candidate. Default=0.
- `num_processes`: Number of processes for parallel processing of OpenAI API calls. Default=8.
- `error_patience`: If the annotator encounters an error, it will retry for `error_patience` times. Default=5.
