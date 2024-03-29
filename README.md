# Machine-Learning-Final-Project

## Setting up a virtual environment
```
# Create a new virtual environment.
python3 -m venv python3-project
# Activate the virtual environment.
source python3-project/bin/activate
# Install packages as specified in requirements.txt.
pip3 install -r reqs.txt
# Optional: Deactivate the virtual environment, returning to your system’s setup.
deactivate
```
## Running the code
### RNN
Train:
```
python3 main_RNN.py --mode "train" --dataDir "datasets" --logDir "log_files" --modelSaveDir "model_files" --LR 0.01 --bs 1 --epochs 362
```
Predict:
```
python3 main_RNN.py --mode "predict" --dataDir "datasets" --weights "model_files/RNN.pt" --predictionsFile "RNN_predictions.csv"
```
### LSTM (not yet implemented)
Train:
```
python3 main_bestmodel.py --mode "train" --dataDir "datasets"
```
### MLP
Train:
```
python3 main_MLP.py --mode "train" --dataDir "datasets" --logDir "log_files" --modelSaveDir "model_files" --LR 0.0001  --bs 10 --epochs 10000
```
Predict:
```
python3 main_MLP.py --mode "predict" --dataDir "datasets" --weights "model_files/MLP.pt" --predictionsFile "MLP_predictions.csv"
```
