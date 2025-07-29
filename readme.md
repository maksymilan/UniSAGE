# SDHyperStructureP
SDHyperStructureP's overview

## Installation
the training data of SDHyperStructureP is referenced from [relbench](https://github.com/snap-stanford/relbench.git), so you need to install it first

## Training
To train the SDHyperStructureP model, you can use the following command:

first, navigate to the `data/generate_data` directory:
```bash
cd /path/to/SDHyperStructureP
cd data/generate_data
```

then, run the script to generate the jsonl format training data:
```bash
./build_dataset.sh
```

after that, to simplify the training process, we sample the training data from the generated jsonl file:
```bash
python data_sampling.py
```

now, we have the training data ready, we will preprocess the data to get pt format data to train the model:
```bash
cd ..
./preprocess.sh
```

finally, we can train the model using the following command:
```bash
cd ..
```

for the classification task:
```bash
./run_classification.sh
```

for the regression task:
```bash
./run_regression.sh
```
