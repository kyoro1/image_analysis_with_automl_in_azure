# image_classification_with_automl_in_azure

This repository provides sample codes, which enable you to learn how to use auto-ml image classification (multi-label) under Azure ML environment.



# Prerequisites:
- Azure subscription
- image files to be used
- 


# How to use
- Please prepare [Azure subscription](https://azure.microsoft.com/en-us/free/).
- 


1) [Provisioning]


# Au

We use Azure ML pipeline as batch execution like deep learning training or inference with this repository. In order to do it, we need `train.py`, which will be embedded in the pipeline.

As a preparation, we need to use Azure ML workspace, and use two kinds of authentication: 1) `az` cli(command line interface) for generating Azure ML pipeline, 2) managed identity for executing Azure ML pipeline.

2) tips for 
- For basic concept of managed identity in generating AML computing cluster, please see [this page](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python#set-up-managed-identity). This repository shows an example for system assigned identity. 
- After generating the identity, please 







# Reference
- [Training an Image Classification Multi-Class model using AutoML](https://github.com/Azure/azureml-examples/blob/main/python-sdk/tutorials/automl-with-azureml/image-classification-multiclass/auto-ml-image-classification-multiclass.ipynb)