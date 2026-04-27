### This file contains notes of this project
##### Created by Toothless7788


## Note
- We are using ```StandardScaler``` for scaling
    - Consider using other scalers in the future
    - ### Example
        - ```RobustScaler```
- If there are more base learners than GPUs available, **round-robin** will be used to distribute processes of training base learner to different GPUs
- You should create model instances in the **spawned process** by ```mp.spawn(...)```
    - If you try to pass the created model instances to the spawned process, unexpected behaviour might occur
    - TODO: Confirm
- ```torch.no_grad()``` and ```<model>.eval()``` are 2 different things and they do not overlap in terms of functionalities
- Using **Inter-Process Communication** (*IPC*), e.g. ```mp.Manager().Queue()```, instead of **Distributed Data Parallel** (*DDP*) for training/evaluation limits the **scability** of the ensemble model
    - *IPC* means you can only have 1 machine (with multiple *processes*)
    - *DDP* means you can have multiple machines (each with multiple *processes*)
    - By the way, using ```mp.Queue()``` instead of ```mp.Manager().Queue()``` gives the error ```Connection refused```
- ```zip(<iterable_1>, <iterable_2>)``` will return a combination of the 2 iterables with length equals to the shorter iterable if the 2 iterables have different lengths
    - There was a bug arised due to mismatch between ```len(layers)``` and ```len(activations)``` in ```mlp.py```
        - It failed silently


## Dataset
- ```housing.csv```
    - ```CRIM```
        - Per capita crime rate by town
    - ```ZN```
        - Proportion of residential land zoned for lots over 25,000 sq.ft.
    - ```INDUS```
        - Proportion of non-retail business acres per town.
    - ```CHAS```
        - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    - ```NOX```
        - Nitric oxides concentration (parts per 10 million)
    - ```RM```
        - Average number of rooms per dwelling
    - ```AGE```
        - Proportion of owner-occupied units built prior to 1940
    - ```DIS```
        - Weighted distances to five Boston employment centres
    - ```RAD```
        - Index of accessibility to radial highways
    - ```TAX```
        - Full-value property-tax rate per $10,000
    - ```PTRATIO```
        - Pupil-teacher ratio by town
    - ```B```
        - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    - ```LSTAT```
        - % lower status of the population
    - ```MEDV```
        - Median value of owner-occupied homes in $1000's


## NEAT
- A **genome** is assigned to a species once its **compatibility score** is **lower** than the **threshold**
    - If the *genome* is compatible to more than 1 species, it will be assigned to the first compatible species
        - In reality, this *genome* might belong to the later species but NEAT does not aim at finding the best match
- If ```node_id``` of the **key** of a **connection** (in ```genome.connections```) is **negative**, it is not an actual node but rather a **bias node**
    - A *bias node* always returns a value of ```1```


## My algorithm
- The **topological sort** of nodes indicates the order of nodes, 1 node for each layer
    - For input nodes, they might not be at the top of the layers
- ### Limitation
    - Activation function must be **idempotent**
    - There must be only 1 output feature
        - Otherwise, some output nodes will be treated as hidden nodes in the new neural network
        - To get the output values, you must extract them by indexing them individually