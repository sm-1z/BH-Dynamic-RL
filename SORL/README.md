## Install

### System

We test on Windows, but still can run on Liniux

### Python

Used python version is `3.8.10`

### Requirement

In order to install the corresponding python package, you can perform the following steps:

```powershell
>>>cd SORL
>>>pip install -r requirement.txt
```

## Structure

`bh_dyn/`contains the BH_Dyn model files

`logs/`contains the tensorboard logs

`models/`contains the train models, you can chooose whether to load the train model

`videos/`contains the videos which are recorded during training

## Start

First, you should execute the following command in order to enter the SORL working directory

```powershell
>>>cd SORL
```

And now, you can execute the command to start to train Humanoid

```powershell
>>>python Humanoid.py
```

## DATA

You can use tensorboard to see the data

```powershell
>>>cd SORL
>>>tensorboard --logdir=logs
```

