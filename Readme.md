# Privacy-Preserving Individual-Level COVID-19 Infection Prediction via Federated Graph Learning

This is the Pytorch implementation of _Falcon_ (a **F**ederated gr**A**ph **L**earning method for privacy-preserving individual-level infe**C**tion predicti**ON**)

![The overall architecture of _Falcon_](./Overview.png)

## Requirements

- torch>=1.11.0
- numpy>=1.23.4
- scikit-learn>=1.1.3
- torch_geometric>=2.1.0
- tensorboard>=2.11.0
- torchinfo>=1.7.1
- tqdm>=4.64.1

Dependency can be installed with the following command:

```bash
pip install -r requirements.txt
```

## Data Preparation
For the consideration of the user privacy, and avoid the xxx usage of mobility data,
the original mobility data . If you need the original mobility dataset for research, please contact with us.

The related files (graph construction files, health status labels, etc.) for **Basic** scenario and **Larger** scenario,
i.e., `/basic` and `/larger`, are available at [Google Drive](), and should be put into the folder `/datasets/beijing`.



## Model Training

Here are commands for training the model on both **Basic** scenario and **Larger** scenario.


```bash
python train.py
```

To train on the different scenarios, please modify the _"dataset"_ item in the config file `config.json`

* ### Basic Scenario

    ```json
      "env_args":
      {
        "train_ratio": 0.4,
        "sim_days": 14,
        "seq_num": 1,
        "unique_len": 16,
        "dataset": "Basic"
      },
    ```

* ### Larger Scenario

    ```json
      "env_args":
      {
        "train_ratio": 0.4,
        "sim_days": 40,
        "seq_num": 1,
        "unique_len": 16,
        "dataset": "Larger"
      },
    ```
  
## Results Visualization

To activate the visualization of experiments, please to check up the item _"tensorboard"_ in `config.json` is set to true, 
then run the following command:
```bash
tensorboard --logdir runs --host 0.0.0.0
```
The visualization results can be found in [http://localhost:6006](http://localhost:6006)