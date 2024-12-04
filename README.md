# CARE
Code for "Leveraging Content and Acoustic Representations for Speech Emotion Recognition"

There are two folders 
* pretraining: consists of the pre-training code. Run **train_pase.py** for this. Remember to store the PASE+ targets and the RoBERTa representations of the ASR transcripts before running this. The paths for these are to be changed in **config.py**
* downstream: consists of storing and then training models for the different downstream datasets 
