import argparse
import configparser
from azureml.core import Workspace, Experiment
from azureml.core.authentication import MsiAuthentication
from azureml.core import Dataset
from azureml.core.compute import AmlCompute

from azureml.automl.core.shared.constants import ImageTask
from azureml.train.automl import AutoMLImageConfig
from azureml.train.hyperdrive import GridParameterSampling, choice
from azureml.core.model import Model


def populate_tabular_dataset(workspace, dataset, dataset_name, train_ratio, train_flg, seed):
    if train_flg:
        tmp_dataset = dataset.random_split(percentage=train_ratio, seed=seed)[0] ## Train data
    else:
        tmp_dataset = dataset.random_split(percentage=train_ratio, seed=seed)[1] ## Test data
    print(f'{dataset_name}: {len(tmp_dataset.to_pandas_dataframe())} records')
    ## Register dataset
    return tmp_dataset.register(name=dataset_name, workspace=workspace, create_new_version=True)


## Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--base_model", help="base_model")
parser.add_argument("--dataset_name", help="dataset_name")
parser.add_argument("--train_ratio", help="train_ratio")
parser.add_argument("--random_seed", help="random seed")
parser.add_argument("--model_name", help="model_name")

args = parser.parse_args()
base_model = args.base_model
dataset_name = args.dataset_name
train_ratio = float(args.train_ratio)
random_seed = int(args.random_seed)
model_name = args.model_name

## Setting
config_ini = configparser.ConfigParser()
config_ini.read('./common/config.ini', encoding='utf-8')

subscription_id = config_ini.get('Azure', 'subscription_id')
resource_group = config_ini.get('Azure', 'resource_group')
workspace_name = config_ini.get('Azure', 'workspace_name')

cluster_name = config_ini.get('AML', 'cluster_name')
train_ratio = config_ini.get('AML', 'train_ratio')

## Authentication with managed identity
msi_auth = MsiAuthentication()

## Retrieve Azure ML workspace
ws = Workspace(subscription_id=subscription_id,
                resource_group=resource_group,
                workspace_name=workspace_name,
               auth=msi_auth)

## Retrieve computer cluster setting
try:
    compute_target = AmlCompute(ws, cluster_name)
    print('Found existing computing cluster.')
except:
    raise Exception(f"Compute {cluster_name} is not found in workspace {ws}.")

## Retrieve experiment
experiment_name = "automl-image-multiclass"
experiment = Experiment(ws, name=experiment_name)

## dataset
dataset = Dataset.get_by_name(ws, name=dataset_name)

## Train/Test split
ds_train_dataset = populate_tabular_dataset(workspace=ws
                                        ,dataset=dataset
                                        ,dataset_name='train_dataset'
                                        ,train_ratio=train_ratio
                                        ,train_flg=True
                                        ,seed=random_seed)

ds_test_dataset = populate_tabular_dataset(workspace=ws
                                        ,dataset=dataset
                                        ,dataset_name='test_dataset'
                                        ,train_ratio=train_ratio
                                        ,train_flg=False
                                        ,seed=random_seed)

## Define image classification task
image_config = AutoMLImageConfig(
    task=ImageTask.IMAGE_CLASSIFICATION,
    compute_target=compute_target,
    training_data=ds_train_dataset,
    validation_data=ds_test_dataset,
    hyperparameter_sampling=GridParameterSampling({"model_name": choice(base_model)}),
    iterations=1,
)

## Execute actual training under remote environment
automl_image_run = experiment.submit(image_config)
automl_image_run.wait_for_completion(wait_post_processing=True)

## Post-process after completing the training
best_child_run = automl_image_run.get_best_child()
## Register ML model associated to AML workspace
model = best_child_run.register_model(
    model_name=model_name
    ,model_path="outputs/model.pt"
)

print('Completely finished!')
