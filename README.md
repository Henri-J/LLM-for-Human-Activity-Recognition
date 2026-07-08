# ML-HAR

This work stems from a final-year research project at Polytech Nantes Engineering School, supervised by a PhD student and professors from the Ls2N laboratory.

The project was carried out with my peers Théo Masselot and Loïc Weber, focusing on the use of Large Language Models (LLMs) as time-series classifiers, specifically for human activity time-series data (HAR).

Originally developed on the university’s GitLab, this is a personal copy shared here for reference.

Our report is is the "Rapport.pdf" file.

# Setup

## Files to download

### Download manually

#### Download the AutoTimes dataset :

generate a link by going here :
<https://cloud.tsinghua.edu.cn/f/0a758154e0d44de890e3/>  

```
wget https://cloud.tsinghua.edu.cn/seafhttp/files/......./dataset.zip 
unzip ./dataset.zip
rm ./dataset.zip
```

#### Download the llama7b models weights :

```
./downloads/download_model.sh
```

## Setup

- Install python >3.10

```
python3 --version
```

- Install all dependencies

```
./setup_venv.sh
```


## Generate the embedding (optional)

They are already in the dataset for (Etth1, Electricity, traffic and weather)

```
python ./preprocess.py
```


