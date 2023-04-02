python main.py --job=training --num_epochs=50 --learning_rate=1e-5 --device=cuda:1
python main.py --job=testing --decoding_strategy=greedy --device=cuda:1
python main.py --job=testing --decoding_strategy=beam --beam_size=5 --device=cuda:1
python main.py --job=testing --decoding_strategy=beam --beam_size=10 --device=cuda:1
python main.py --job=testing --decoding_strategy=beam --beam_size=20 --device=cuda:1
