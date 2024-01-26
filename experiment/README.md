# [EACL 2024 Findings] GPTs are (Multilingual) Annotators for Sequence Generation Tasks

## Experiment - How to start

```shell
$ conda create -n proj-gpt-annotator python=3.8
$ conda activate proj-gpt-annotator
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install -r requirements.txt
$ bash run_captioning.sh
$ bash run_text_style_transfer.sh
```
