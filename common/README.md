Please prepare `config.ini` under this directory (`/common`) with the following format. They will be used in the scripts like `train.py`, generating AML pipeline, etc.

You can find two files as a sample: 
- `config_ic.ini`: for image classification
- `config_od.ini`: for object detection

```config.ini
[Azure]
subscription_id=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
resource_group=[YOUR RESOURCE GROUP]
workspace_name=[YOUR AML WORKSPACE NAME]

[data]
train_ratio=0.8
dataset_name=weather-classification_20220613_131039
dataset_name_for_train=train_dataset
dataset_name_for_test=test_dataset

[AML]
## general settings
cluster_name=gpu-cluster-nc6
vm_size=Standard_NC6
vm_location=australiaeast
managed_id=SystemAssigned
random_seed=1

## For Image Classification
#image_analysis_type=image_classification
#experiment_name=automl-image-multiclass
#base_model=vitb16r224
#pipelineName=image_classification_pipeline
#model_name=image_classification_automl

## For Object Detection
#image_analysis_type=object_detection
#experiment_name=automl-object-detection
#base_model=yolov5
#pipelineName=object_detection_pipeline
#model_name=object_detection_automl_kyiwasak

## For inference
experiment_name_for_inference=automl_inference
pipelineName_for_inference=inference_pipeline
```

`Azure` part: Basic Azure parameters

| variable        | description                    |
------------------|--------------------------------|
| subscription_id | Your subscription ID of Azure  |
| resource_group  | Your resource group            |
| workspace_name  | Your AML workspace name        |

`data` part: Basic data parameters

| variable                   | description                                                                  |
-----------------------------|------------------------------------------------------------------------------|
| train_ratio                | If you specify `0.8`, 80% image files for training, and others for testing.  |
| dataset_name               | Your dataset name in AML workspace, which are exported after data labelling. |
| dataset_name_for_train     | Subset of dataset for training process                                       |
| dataset_name_for_test      | Subset of dataset for testing process                                        |


`AML` part: Used for populating AML resources

| variable                      | description                                                          |
--------------------------------|----------------------------------------------------------------------|
| cluster_name                  | Name of computer cluster used in image classification.               |
| vm_size                       | Specify VM used in training of ML model[^1]                          |
| vm_location                   | Specify location of VM                                               |
| managed_id                    | This id is used for authenticating in training under AML pipeline    |
| random_seed                   | Used in splitting train/test data                                    |
| image_analysis_type           | Select analysis type from `image_classification`, `object_detection` |
| experiment_name               | Experiment name, a group of jobs                                     |
| base_model                    | Base image classification model[^2]                                  |
| pipelineName                  | Name of AML pipeline                                                 |
| model_name                    | Name of model, used both in training/inferencing.                    |
| experiment_name_for_inference | experiment name for inferrence                                       |
| pipelineName_for_inference    | pipeline name for inferrence                                         |

[^1]: Please specify NC-series or ND-series instead of NV-series
[^2]: You can change it from [the list here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-image-models?tabs=CLI-v2#supported-model-algorithms)

