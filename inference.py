import argparse
from azureml.core import Workspace
from azureml.core.authentication import MsiAuthentication
from azureml.core.compute import AmlCompute
from azureml.core.model import Model
from azureml.automl.dnn.vision.common.model_export_utils import load_model, run_inference

## Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--subscription_id", help="subscription id")
parser.add_argument("--resource_group", help="resource group")
parser.add_argument("--workspace_name", help="AML workspace")
parser.add_argument("--cluster_name", help="cluster name")
parser.add_argument("--model_name", help="model name")
parser.add_argument("--TASK_TYPE", help="image analysis type")

args = parser.parse_args()
subscription_id = args.subscription_id
resource_group = args.resource_group
workspace_name = args.workspace_name
cluster_name = args.cluster_name
model_name = args.model_name
TASK_TYPE = args.TASK_TYPE

## Authentication
msi_auth = MsiAuthentication()

## Retrieve workspace
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


## Distinguish parameters for inference
if TASK_TYPE == 'image-classification':
    from azureml.automl.dnn.vision.classification.inference.score import _score_with_model
elif TASK_TYPE == 'image-object-detection':
    from azureml.automl.dnn.vision.object_detection_yolo.writers.score import _score_with_model

model_path = Model.get_model_path(model_name=model_name, _workspace=ws)
ml_model = load_model(TASK_TYPE, model_path)

image = open('./sample_images/60_1537642922.jpg', 'rb').read()

result = run_inference(ml_model, image, _score_with_model)
print(result)


