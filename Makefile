# Convenience commands
PY=python

data:
	$(PY) data_prep.py --data_dir ./data --download --spm_vocab_size 8000

train-sentiment:
	$(PY) train.py train_sentiment --data_dir ./data --spm_model ./spm/bpe8k.model --out_dir ./outputs --model_type transformer

train-lyrics:
	$(PY) train.py train_lyrics --data_dir ./data --spm_model ./spm/bpe8k.model --out_dir ./outputs --model_type transformer

train-music:
	$(PY) train.py train_music --data_dir ./data --out_dir ./outputs --model_type transformer

infer:
	$(PY) train.py infer --input_text "오늘 하루 좀 힘들었어" \	  --sentiment_lstm ./outputs/sentiment_lstm.pth \	  --sentiment_tr   ./outputs/sentiment_transformer.pth \	  --lyrics_lstm    ./outputs/lyrics_lstm.pth \	  --lyrics_tr      ./outputs/lyrics_transformer.pth \	  --music_lstm     ./outputs/music_lstm.pth \	  --music_tr       ./outputs/music_transformer.pth \	  --out_dir ./outputs
