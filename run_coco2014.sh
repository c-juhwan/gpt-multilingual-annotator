DEVICE=cuda:1
DATA=coco2014

clear
#python main.py --job=preprocessing --task_dataset=${DATA}
#python main.py --task=annotating --job=backtrans_annotating --task_dataset=${DATA}
#python main.py --task=annotating --job=gpt_annotating --task_dataset=${DATA}

#python main.py --job=training --task_dataset=${DATA} --annotation_mode=aihub_ko --num_epochs=50 --learning_rate=5e-5 --device=${DEVICE} --desc="Full_Gold_T" --log_freq=5000
#python main.py --job=testing  --task_dataset=${DATA} --annotation_mode=aihub_ko --decoding_strategy=greedy --device=${DEVICE} --desc="Full_Gold_G"
#python main.py --job=testing  --task_dataset=${DATA} --annotation_mode=aihub_ko --decoding_strategy=beam --beam_size=10 --device=${DEVICE} --desc="Full_Gold_B10"

python main.py --job=training --task_dataset=${DATA} --annotation_mode=original_en --num_epochs=50 --learning_rate=5e-5 --device=${DEVICE} --desc="Full_Gold_T" --log_freq=5000
python main.py --job=testing  --task_dataset=${DATA} --annotation_mode=original_en --decoding_strategy=greedy --device=${DEVICE} --desc="Full_Gold_G"
python main.py --job=testing  --task_dataset=${DATA} --annotation_mode=original_en --decoding_strategy=beam --beam_size=10 --device=${DEVICE} --desc="Full_Gold_B10"

#python main.py --job=training --task_dataset=${DATA} --annotation_mode=gpt_en --num_epochs=50 --learning_rate=5e-5 --device=${DEVICE} --desc="GPT_Silver_T"
#python main.py --job=testing  --task_dataset=${DATA} --annotation_mode=gpt_en --decoding_strategy=greedy --device=${DEVICE} --desc="GPT_Silver_G"
#python main.py --job=testing  --task_dataset=${DATA} --annotation_mode=gpt_en --decoding_strategy=beam --beam_size=10 --device=${DEVICE} --desc="GPT_Silver_B10"

#python main.py --job=training --task_dataset=${DATA} --annotation_mode=backtrans_en --num_epochs=50 --learning_rate=5e-5 --device=${DEVICE} --desc="BT_Bronze_T"
#python main.py --job=testing  --task_dataset=${DATA} --annotation_mode=backtrans_en --decoding_strategy=greedy --device=${DEVICE} --desc="BT_Bronze_G"
#python main.py --job=testing  --task_dataset=${DATA} --annotation_mode=backtrans_en --decoding_strategy=beam --beam_size=10 --device=${DEVICE} --desc="BT_Bronze_B10"
