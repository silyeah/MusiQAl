# MusiQAl: Music Question-Answering through Audio-Video Fusion

# Description

The MusiQAl Dataset is a comprehensive collection designed to advance the field of music question-answering (MQA). It features 310 carefully selected videos and 11,793 question-answer pairs, totaling five hours of music performance footage. Performances are sourced from notable projects and public datasets, along with a diverse range of YouTube videos.

## Content and Representation
47 Music Instruments: Spanning a wide range of musical traditions and genres.
Performances Across 18 Countries and Five Continents: Showcasing a global perspective on music.
11 Question Types: Covering audio, visual, and audio-visual scenarios.
The dataset highlights four specific aspects of music performance: instrument playing, dancing, singing, and their combinations. Recognizing the cultural significance of rhythm and dance, MusiQAl includes music made for dance, integral to many traditions.

## Data Availability
For access to the dataset, please visit this link: https://zenodo.org/records/13623449

# Requirements

```bash 
pip install -r compat_requirements.txt
```

# Usage

## Download data

Please make sure to download the wav files, video frames, features, and JSON files from this link: https://zenodo.org/records/13623449

## Train AVST model on complete dataset

Run [train.sh](https://github.com/silyeah/MusiQAl/blob/in5490/AVST/train.sh)  


## Train AVST model on ablated dataset

Define interventions in the parser arguments in [intv_train_avst.py](https://github.com/silyeah/MusiQAl/blob/in5490/AVST/net_grd_avst/intv_train_avst.py). 

Run [intv_train.sh](https://github.com/silyeah/MusiQAl/blob/in5490/AVST/intv_train.sh)  


## Test AVST model on complete or ablated dataset

Define which python script to run, based on desired intervention. See [test.sh](https://github.com/silyeah/MusiQAl/blob/in5490/AVST/test.sh) for examples. 

Run [test.sh](https://github.com/silyeah/MusiQAl/blob/in5490/AVST/test.sh)  



