<h2 align="center">
  <b>Buffer-GOPT</b>

</h2>

## Installation

```bash
git clone https://github.com/PHBach123/Buffer-GOPT.git
cd GOPT

conda create -n GOPT python=3.9
conda activate GOPT

# install pytorch
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# install other dependencies
pip install -r requirements.txt
```

## Training
The dataset is generated on the fly, so you can directly train the model by running the following command.

```bash
python ts_train.py --config cfg/config.yaml --device 0 
```

If you do not use the default dataset (the bin is 10x10x10), you can modify the tag `env` in `cfg/config.yaml` file to specify the bin size and the number of items.
Note that most hyperparameters are in the `cfg/config.yaml` file, you can modify them to fit your needs.


## Evaluation

```bash
python ts_test.py --config cfg/config.yaml --device 0 --ckp /path/to/policy_step_final.pth
```

If you want to visualize the packing process of one test, you can add the `--render` flag.
```bash
python ts_test.py --config cfg/config.yaml --device 0 --ckp /path/to/policy_step_final.pth --render
```

