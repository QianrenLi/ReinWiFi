# ReinWiFi

A reninfocement-learning-based application-layer QoS optimization framework of WiFi Networks.

## Introduction

This is the code repository for the ReinWiFi project. The project aims to optimize the application-layer QoS of WiFi networks using Reinforcement Learning (RL) techniques. The project is developed in Python and uses the PyTorch library for the RL implementation. The project is still in its early stages and is under development.

The controller is specifically designed to handle the application-layer QoS of the WiFi network as following shows.
![controller](./figs/network.png "controller")

## Usage

Online Testing for proposed model:

```bash
python main.py --config config/modelTest.yaml
```

Offline Training for proposed model for imitator and controller, repspectively:

```bash
python3 train_from_his.py --config config/trainHisEnv.yaml
python3 train_from_his.py --config config/trainHisCtl.yaml
```

Online Training for controller:

```bash
python main.py --config config/modelTrain.yaml
```

## TODO List

- [ ] Update README.md
- [ ] Installation guide, with requirements and scripts provided.
- [ ] Code Cleanup

## Acknowledgement
This project is build on top of the [stream-replay](https://github.com/lasso-sustech/stream-replay) and [wlsops-hack](https://github.com/lasso-sustech/wlsops-hack).
