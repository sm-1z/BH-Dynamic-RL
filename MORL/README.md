## Install

### System

We test on Windows, but still can run on Liniux

### Python

Used python version is `3.8.10`

### Requirement

In order to install the corresponding python package, you can perform the following steps:

```powershell
>>>cd MORL
>>>pip install -r requirement.txt
```

**P.S.**

You can clear python packages with the following command:

```powershell
>>>cd MORL
>>>pip freeze > deltemp.txt
>>>pip uninstall -r deltemp.txt -y
```

## Structure

`bh_dyn/`contains the BH_Dyn model files

`logs/`contains the tensorboard logs

`models/`contains the train models, you can chooose whether to load the train model

`videos/`contains the videos which are recorded during training



## Start

First, you can use tools like `venv`, `conda`, etc. to create a new python virtual environment in which to install python packages. We use `conda` to create a new python environment and start training. In order to create a new environment with conda, make sure that conda is installed on your system and can be created with the following command:

```powershell
>>>conda create -n myMORL python=3.8
>>>conda init
>>>conda activate myMORL
```

Now, you activate a python virtual environment, and you can start to train the model. First, you should execute the following command in order to enter the MORL working directory:

```powershell
>>>cd MORL
```

Then, you can execute the command to start to train Humanoid

```powershell
>>>python Humanoid.py
```

```powershell
>>>python bhmodel_train.py
```

## DATA

You can use tensorboard to see the data

```powershell
>>>cd MORL
>>>tensorboard --logdir=logs
```

You also can test the model:

```powershell
>>>cd MORL
>>>python bhmodel_test.py
```

## Final

You can change the training environment by copying the code and modifying the variables when generating the gym environment.