Please prepare `config.ini` under this directory (`/common`) with the following format. They will be used in the scripts like `train.py`, generating AML pipeline, etc.

```config.ini
[Azure]
subscription_id=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
resource_group=[YOUR RESOURCE GROUP]
workspace_name=[YOUR AML WORKSPACE NAME]

[data]
train_ratio=0.8
dataset_name=weather-classification_20220613_131039

[AML]
cluster_name=gpu-cluster-nc6
vm_size=Standard_NC6
vm_location=australiaeast
managed_id=SystemAssigned
base_model=vitb16r224
random_seed=1
pipelineName=image_classification_pipeline
model_name=image_classification_automl
```

`Azure` part: Basic Azure parameters

| value           | description                    |
------------------|--------------------------------|
| subscription_id | Your subscription ID of Azure  |
| resource_group  | Your resource group            |
| workspace_name  | Your AML workspace name        |

`data` part: Basic data parameters

| value           | description                                                     |
------------------|-----------------------------------------------------------------|
| train_ratio     | If `0.8`, 80% image files for training, and others for testing. |
| dataset_name    | Your dataset name in AML workspace, which are exported after data labelling. |

`AML` part: Used for populating AML resources

| value          | description                                    |
-----------------|-------------------------------------------------------------------|
| cluster_name   | Name of computer cluster used in image classification.            |
| vm_size        | Specify VM used in training of ML model[^1]                       |
| vm_location    | Specify location of VM                                            |
| managed_id     | This id is used for authenticating in training under AML pipeline |
| base_model     | Base image classification model[^2]                               |
| random_seed    | Used in splitting train/test data                                 |
| pipelineName   | Name of AML pipeline                                              |
| model_name     | Name of model, used both in training/inferencing.                 |


[^1]: Please specify NC-series or ND-series instead of NV-series
[^2]: You can change it from [the list here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-image-models?tabs=CLI-v2#supported-model-algorithms)

