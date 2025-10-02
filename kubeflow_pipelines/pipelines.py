import kfp
from kfp import dsl, compiler
from kfp.dsl import component

@component(
    base_image="vijaytakbhate1/medical_insurance_cost_prediction_svr:latest"
)
def data_processing_op():
    import subprocess
    subprocess.run(["python", "processing_pipeline_runner.py"], check=True)


@component(
    base_image="vijaytakbhate1/medical_insurance_cost_prediction_svr:latest"
)
def model_training_op():
    import subprocess
    subprocess.run(["python", "training_pipeline_runner.py"], check=True)


@dsl.pipeline(
    name="medical_insurance_cost_prediction_app",
    description="This pipeline will process the data and train the model"
)
def mlops_pipeline():
    data_processing = data_processing_op()
    model_training = model_training_op().after(data_processing)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=mlops_pipeline,
        package_path="kubeflow_pipelines/Insurance_Cost_Prediction_pipeline_02_10_25.yaml"
    )
