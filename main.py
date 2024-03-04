from SageMakerWrapper.data import upload_data
from SageMakerWrapper.training import launch_training

#upload_data(config_dir="config", data_dir="data")

launch_training(code_dir="code", config_dir="config")
