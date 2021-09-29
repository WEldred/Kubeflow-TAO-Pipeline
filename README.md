# Kubeflow TAO (3.0) Pipeline Example

## Overview

NVIDIA released sample code demonstrating how to build a Kubeflow pipeline for a NVIDIA TLT workflow (transfer learning).  The sample was built for version 1.0 of TLT.  The goal was to demonstrate how to build the pipeline and the sample showed how using Transfer Learning with a Object Detection pipeline.  The process to build this pipeline is very manual and the example exposed a limited setup of TLT commands necessary.  It didn't expose all possible operators or parameters that could be passed to calls.  Again - a simple blue print on how someone might build their own pipeline.

Just recently, NVIDIA has release TAO (version 3.0) which is the next version of TLT.  Several customers have asked about building pipelines for TAO in Kubeflow.  This repository is an example of building a "Classification" pipeline based on the [Classification Notebook Example](https://docs.nvidia.com/tao/tao-toolkit/text/tao_toolkit_quick_start_guide.html#cv-samples).  

This is not intended to be a full project, but rather a sample pipeline conversion that you can use as a basis to create your own pipelines. 

## General Guidance

My general guidance for this example is to first build a functioning workflow from the command line and then use that as a basis for creating your own pipeline script.

## Pipeline Creation Process

There are several key components used in building a pipeline:
- **Component Container(s)**: These are containers with applications that are run by the pipeline.  For my entire pipeline, I used the standard TAO NGC container as I needed to run TAO commands against my models and data files.
- **Pipeline Python Files**: The files are the ones you'll create to build your pipeline.
- **Pipeline YAML File**: This file is created by the Python files and is uploaded to Kubeflow to add the pipeline.

For the "Classification Example" here, there are 2 key files used to build the pipeline:
- **tao_iva_classification_pipeline.py**: This file defines the "parameters" that are input to the pipeline and the DAG for the workflow (that calls operators in the 'ops' file)
- **tao_iva_classification_ops.py**: This file defines all the operators that are called from the DAG.  The operators define all the TAO command line calls and parameters.

When the **tao_iva_classification_pipeline.py** file is run, it will create the "YAML" file that is uploaded into Kubeflow.

The actual steps for creating are:
1. You'll need an environment with Python3
2. You'll need to install the Kubeflow Pipelines SDK: `pip3 install kfp --upgrade`
3. Edit the "pipeline.py" file to build out the graph for the sequence of commands you wish to run.  You may need to update the "ops.py" file as it only exposes a subset of all possible TAO operations and passed parameters.
4. Run the "pipeline.py" file to create the "pipeline.yaml" file: `python3 tao_iva_classification_pipeline.py`
5. Take the generated YAML file and upload it into Kubeflow.  You'll go to the "Pipelines" section and choose "Add Pipeline".

## Key Issues To Be Aware Of When Building Pipeline

Here are some of the issues I dealt with:
- This pipeline assumes that there is a "volume" (could be K8s PV, NFS, etc) mounted that contains all of the models, training data, and newly created models.  For this example, I used "/mnt/workspace" as the mount point for my key files.  I had this pre-mounted in my environment - you'll have to find a strategy that works for your environment.  (I didn't see an easy way to do it in the pipeline and left it as a future exercise)
- Each of the TAO models (classification, object detection, ...) have their on data file formats and conversions required.  You'll have to review the [documentation](https://docs.nvidia.com/tao/tao-toolkit/text/data_annotation_format.html) and make the appropriate changes for different types of models.
- Each of the TAO models has it's own specification files and requirements.  This is a large part of the configuration for the training and evaluation.  You'll need to create your own versions for other models.
- In general, because of all the variations of files and configuration, there is not a clean way to create one interface.  So, realistically, you'll want to build different pipelines for different models.
- In general, while you could pass many parameters through the Kubeflow Pipeline Create Run interface, it only presents basic text input boxes so you wouldn't want to have too many on the screen.
 
## Example Classification Pipeline

Here's the command sequence from the example that I converted into a pipeline:
1. command 1
2. command 2
3. command 3

## Example Classification Directory Layout
- /mnt/workspace: Main mount point for my data science volume
 - models: Location to download NGC Pre-Trained Models to.
 - specs: Specification files used for calls to `classification Train` and `classification evaluate` (for both training and retraining)
 - tao-experments: Location of key files for training
  - data: Customer data for fine-tuning models.  
  - export: Location for output model files
  - output: Need to Check
  - output-pruned: Need to check
  - output-retrained: Need to check

