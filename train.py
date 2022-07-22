import argparse
from azureml.core import Workspace, Experiment, Dataset
from azureml.core.authentication import MsiAuthentication
from azureml.core.compute import AmlCompute

from azureml.automl.core.shared.constants import ImageTask
from azureml.train.automl import AutoMLImageConfig
from azureml.train.hyperdrive import GridParameterSampling, choice


def populate_tabular_dataset(workspace, dataset, dataset_name, train_ratio, train_flg, seed):
    print(f'Split dataset. Train:Test = {train_ratio}:{1-train_ratio}')
    if train_flg:
        tmp_dataset = dataset.random_split(percentage=train_ratio, seed=seed)[0] ## Train data
    else:
        tmp_dataset = dataset.random_split(percentage=train_ratio, seed=seed)[1] ## Test data
    print(f'{dataset_name}: {len(tmp_dataset.to_pandas_dataframe())} records')
    ## Register dataset
    return tmp_dataset.register(name=dataset_name, workspace=workspace, create_new_version=True)


## Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--subscription_id", help="subscription id")
parser.add_argument("--resource_group", help="resource group")
parser.add_argument("--workspace_name", help="AML workspace")
parser.add_argument("--cluster_name", help="cluster name")
parser.add_argument("--experiment_name", help="experiment name")
parser.add_argument("--base_model", help="base model")
parser.add_argument("--dataset_name", help="dataset name")
parser.add_argument("--dataset_name_for_train", help="dataset name for training")
parser.add_argument("--dataset_name_for_test", help="dataset name for testing")
parser.add_argument("--train_ratio", help="train ratio")
parser.add_argument("--random_seed", help="random seed")
parser.add_argument("--model_name", help="model name")
parser.add_argument("--image_analysis_type", help="image analysis type")

args = parser.parse_args()
subscription_id = args.subscription_id
resource_group = args.resource_group
workspace_name = args.workspace_name
cluster_name = args.cluster_name
experiment_name = args.experiment_name
base_model = args.base_model
dataset_name = args.dataset_name
dataset_name_for_train = args.dataset_name_for_train
dataset_name_for_test = args.dataset_name_for_test
train_ratio = float(args.train_ratio)
random_seed = int(args.random_seed)
model_name = args.model_name
image_analysis_type = args.image_analysis_type

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

## dataset
dataset = Dataset.get_by_name(ws, name=dataset_name)

## Train/Test split
ds_train_dataset = populate_tabular_dataset(workspace=ws
                                        ,dataset=dataset
                                        ,dataset_name=dataset_name_for_train
                                        ,train_ratio=train_ratio
                                        ,train_flg=True
                                        ,seed=random_seed)

ds_test_dataset = populate_tabular_dataset(workspace=ws
                                        ,dataset=dataset
                                        ,dataset_name=dataset_name_for_test
                                        ,train_ratio=train_ratio
                                        ,train_flg=False
                                        ,seed=random_seed)

## Define image classification task
task = None
if image_analysis_type == 'image_classification':
    task = ImageTask.IMAGE_CLASSIFICATION
elif image_analysis_type == 'object_detection':
    task = ImageTask.IMAGE_OBJECT_DETECTION

if task != None:
    experiment = Experiment(ws, name=experiment_name)
    image_config = AutoMLImageConfig(
        task=task,
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
else:
    print('Please check your setting.')
