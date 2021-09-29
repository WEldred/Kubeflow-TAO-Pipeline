import tao_iva_ops
import kfp.dsl as dsl
from kubernetes import client as k8s_client

@dsl.pipeline(
  name='taoPipeline',
  description='TAO Pipeline Kubeflow Demo'
)
def taoPipeline(
  pretrained_model_name: str ="nvidia/tao/pretrained_classification:resnet18",
  pretrained_model_dir: str ="models",
  train_classification_data: str = "tao-experiments/data",
  train_classification_spec: str = "specs/classification_spec.cfg",
  train_classification_output: str = "tao-experiments/output",
  train_num_gpus: str = "1",
  prune_model_input_file = "tao-experiments/output/weights/resnet_020.tlt",
  prune_model_output_file = "tao-experiments/output-pruned/resnet18_nopool_bn_pruned.tlt",
  prune_equalization_criterion = "union",
  prune_threshold = "0.6",
  retrain_classification_spec: str = "specs/classification_retrain_spec.cfg",
  retrain_classification_output: str = "tao-experiments/output-retrain",
  test_infer_model_input_file: str = "tao-experiments/output-retrain/weights/resnet_020.tlt",
  test_infer_image_path: str = "tao-experiments/data/split/test/person",
  test_infer_class_map: str = "tao-experiments/output-retrain/classmap.json",
  test_infer_batch_size: str = "16",
  export_model_input_file: str = "tao-experiments/output-retrain/weights/resnet_020.tlt",
  export_model_output_file: str = "tao-experiments/export/final_model.etlt",
  calibrate_classification_spec: str = "specs/classification_retrain_spec.cfg",
  calibrate_max_batches: str = "10",
  calibrate_output_file: str = "tao-experiments/export/calibration.tensor",
  export_int8_cache_file: str = "tao-experiments/export/final_model_int8_cache.bin",
  export_int8_batches: str = "10",
  export_int8_model_output_file: str = "tao-experiments/export/final_int8_model.etlt",
  convert_trt_output_file: str = "tao-experiments/export/final_model.trt",
  convert_trt_output_layer: str = "predictions/Softmax",
  convert_trt_dims: str = "3,224,224",
  convert_trt_input_type: str = "nchw",
  convert_trt_max_batch_size: str = "64",
  convert_trt_precision: str = "int8",
  convert_trt_batch_size: str = "64",
  persistent_volume_path: str ="/mnt/workspace",
  api_key: str = "nvidia_tlt",
  ):

  # define some variables
  op_dict = {}
  tao_mount_dir='/mnt/workspace'
  persistent_volume_name='nvidia-workspace'

  # Defining the main pipeline here.

# add component to download pretrained model

  op_dict['tao_pull'] = tao_iva_ops.TAOPullOp(
                          "download-model", tao_mount_dir, pretrained_model_dir, pretrained_model_name)

# add component to train model

  op_dict['tao_train'] = tao_iva_ops.TAOTrainClassificationOp(
                          "train-model", tao_mount_dir, api_key, train_classification_output,
                          train_classification_spec, train_num_gpus)
  op_dict['tao_train'].after(op_dict['tao_pull'])

# add component to evaluate model

  op_dict['tao_evaluate'] = tao_iva_ops.TAOEvaluateClassificationOp(
                          "evaluate-model", tao_mount_dir, api_key, train_classification_spec)
  op_dict['tao_evaluate'].after(op_dict['tao_train'])

# add component to prune model

  op_dict['tao_prune'] = tao_iva_ops.TAOPruneOp(
                          "prune-model", tao_mount_dir, api_key, prune_model_input_file, prune_model_output_file, prune_threshold, prune_equalization_criterion)
  op_dict['tao_prune'].after(op_dict['tao_train'])

# add component to retrain after evaluation and pruning

  op_dict['tao_retrain'] = tao_iva_ops.TAOTrainClassificationOp(
                          "retrain-model", tao_mount_dir, api_key, retrain_classification_output,
                          retrain_classification_spec, train_num_gpus)
  op_dict['tao_retrain'].after(op_dict['tao_prune'])
  op_dict['tao_retrain'].after(op_dict['tao_evaluate'])

# add component to evaluate model

  op_dict['tao_reevaluate'] = tao_iva_ops.TAOEvaluateClassificationOp(
    "reevaluate-model", tao_mount_dir, api_key, retrain_classification_spec)
  op_dict['tao_reevaluate'].after(op_dict['tao_retrain'])

# add component to test inference

  op_dict['tao_test_infer'] = tao_iva_ops.TAOInferenceClassificationOp(
    "test-infer", tao_mount_dir, api_key, test_infer_model_input_file, test_infer_image_path, test_infer_batch_size, test_infer_class_map, retrain_classification_spec)
  op_dict['tao_test_infer'].after(op_dict['tao_reevaluate'])

# export model

  op_dict['tao_export_model'] = tao_iva_ops.TAOExportClassificationOp(
    "export-model", tao_mount_dir, api_key, export_model_input_file, export_model_output_file)
  op_dict['tao_export_model'].after(op_dict['tao_test_infer'])

# calibrate model

  op_dict['tao_calibrate_model'] = tao_iva_ops.TAOCalibrateClassificationOp(
    "calibrate-model", tao_mount_dir, calibrate_classification_spec, calibrate_max_batches, calibrate_output_file)
  op_dict['tao_calibrate_model'].after(op_dict['tao_export_model'])

# export int8 model
  op_dict['tao_export_int8_model'] = tao_iva_ops.TAOExportClassificationAdvancedOp(
    "export-int8-model", tao_mount_dir, api_key, export_model_input_file, export_int8_model_output_file, calibrate_output_file, convert_trt_precision, export_int8_batches, export_int8_cache_file)
  op_dict['tao_export_int8_model'].after(op_dict['tao_calibrate_model'])

# Convert to TRT Engine

  op_dict['tao_create_trt'] = tao_iva_ops.TAOConvertTRTCalibrateOp(
    "convert-trt", tao_mount_dir, api_key, export_int8_model_output_file, export_int8_cache_file, convert_trt_output_file,
    convert_trt_output_layer, convert_trt_dims, convert_trt_input_type, convert_trt_max_batch_size, convert_trt_precision, convert_trt_batch_size)
  op_dict['tao_create_trt'].after(op_dict['tao_export_int8_model'])

  # add volume mount to all components

  for name, container_op in op_dict.items():
    container_op.add_volume(k8s_client.V1Volume(
      host_path=k8s_client.V1HostPathVolumeSource(path=tao_mount_dir.__str__()),
      name=persistent_volume_name))
    container_op.add_volume_mount(k8s_client.V1VolumeMount(
      mount_path=tao_mount_dir.__str__(),
      name=persistent_volume_name))

if __name__ == '__main__':
  import kfp.compiler as compiler
#  compiler.Compiler().compile(workflow1, __file__ + '.tar.gz')
  compiler.Compiler().compile(taoPipeline, __file__ + '.tar.gz')
