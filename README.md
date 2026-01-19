# Compiler for Distributed Quantum Computing: a Reinforcement Learning Approach

This repository contains a Compiler for Distributed Quantum Computing (DQC) that can be trained for real-time compilation of quantum circuits, using different Reinforcement Learning (RL) methods. The compiler is based on the paper "Compiler for Distributed Quantum Computing: a Reinforcement Learning Approach".

This repository is a fork of the original implementation, which in principle supports multiple RL agents (e.g., DQN, DDQN, PPO). This fork however introduces a constrained action-selection mechanism using state-based action masks and other associated modifications (see commits). **Any introduced changes were tested with the DDQN agent only**; other agents (DQN/PPO/etc.) have not been verified with these modifications and may require adjustments. (The basic unconstrained RL methods that serve as a basis for the original implementations are obtained from https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch) 

**QuantumEnvUpdater** is an interface between the RL agent and the compiler. It interacts with the environment every time it is needed - one time at the initialization and once in every episode. **QuantumEnvClass** holds the different parts of the compiler, it creates for every episode (i) the DQC architecture (by creating an instance of **QPUClass()**), and (ii) the circuit to be executed (**DAGClass()**). In the current version of the code, the DQC architecture used is two IBM Q Guadalupe quantum processors connected through a quantum link. Moreover, the **DAGClass()** generates a random circuit with 30 gates. Therefore, the compiler is being trained with a different quantum circuit in every episode. The constants are contained in **Constants**. **QubitMappingClass()** creates mapping objects that treat physical qubits as a box and logical qubits as balls in order to keep the state of the QPU.

## How to run
Pretrained models are saved inside `Models/`
To run inference with a trained model, edit the filepath and ensure correct hyperparameters inside results/DistCompiler.py, add the mapping and circuit to a dataset, edit the filepath to the dataset and set NUMG to the amount of gates the agent was trained on inside Constant.py and then run:
```
python results/DistQuantum.py
```

To run the compiler training process, first edit the DRL method and hyperparamters (optional) inside results/DistQuantum.py and Constants.py, and then run:
```
python results/DistQuantum.py
```
Note that this might take several hours to complete.