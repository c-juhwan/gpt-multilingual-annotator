python main.py --task=annotating --job=gpt_annotating
python main.py --job=training --annotation_mode=gpt_en --num_epochs=50 --learning_rate=1e-5 --device=cuda:0 --desc="500 gold + 2000 silver"
python main.py --job=testing --decoding_strategy=greedy --device=cuda:0 --desc="500 gold + 2000 silver"
python main.py --job=testing --decoding_strategy=beam --beam_size=5 --device=cuda:0 --desc="500 gold + 2000 silver"
python main.py --job=testing --decoding_strategy=beam --beam_size=10 --device=cuda:0 --desc="500 gold + 2000 silver"
python main.py --job=testing --decoding_strategy=beam --beam_size=20 --device=cuda:0 --desc="500 gold + 2000 silver"
