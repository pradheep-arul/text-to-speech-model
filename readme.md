
## Setup
```
python3 -m venv .venv

source .venv/bin/activate

pip install torch torchaudio librosa matplotlib numpy tqdm soundfile pandas
pip install black isort

pip freeze > requirements.txt
```


## GPU Setup
```
# After SSH connection
git clone https://github.com/pradheep-arul/text-to-speech-model.git
cd text-to-speech-model

# Setup Python environment
pip install torch torchaudio librosa matplotlib numpy tqdm soundfile pandas psutil

# Download and extract dataset
mkdir data checkpoints output

cd data
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjf LJSpeech-1.1.tar.bz2
cd ..
```


## Training
```
python src/preprocess_dataset.py
python src/train.py
```


## Inference
```
python src/inference.py
```


