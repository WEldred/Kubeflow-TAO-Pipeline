import kfp.dsl as dsl
from kubernetes import client as k8s_client

__TAO_CONTAINER_VERSION__='weldred-kftao:0.1'

#
# General Structures and Operators
#

class ObjectDict(dict):
  def __getattr__(self, name):
    if name in self:
      return self[name]
    else:
      raise AttributeError("No such attribute: " + name)

class TAORunCommandOp(dsl.ContainerOp):
  def __init__(self, name, command, args):
    super(TAORunCommandOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=[command],
      arguments=[args],
      file_outputs={}
      )

#
# Pre-Trained Model Operators
#

class TAOPullOp(dsl.ContainerOp):
  def __init__(self, name, tao_mount_dir, model_dir, model_name):
    super(TAOPullOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['/opt/ngccli/ngc'],
      arguments=['registry',
                 'model',
                 'download-version',
                 model_name,
                 '--dest', '%s/%s' % (tao_mount_dir, model_dir)
      ],
      file_outputs={}
      )

#
# Classification Operators
#

class TAODatasetConvertOp(dsl.ContainerOp):
  def __init__(self, name, tao_mount_dir, export_spec_path, tfrecords_path):
    super(TAODatasetConvertOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['tao-dataset-convert'],
      arguments=[
        '-d', '%s/%s' % (tao_mount_dir, export_spec_path),
        '-o', '%s/%s' % (tao_mount_dir, tfrecords_path)
      ],
      file_outputs={}
      )
    name=name

class TAOTrainClassificationOp(dsl.ContainerOp):
  def __init__(self, name, tao_mount_dir, api_key, results_dir, spec_file, num_gpus):
    super(TAOTrainClassificationOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['classification'],
      arguments=[
        'train',
        '--gpus', num_gpus,
        '-k', api_key,
        '-e', '%s/%s' % (tao_mount_dir, spec_file),
        '-r', '%s/%s' % (tao_mount_dir, results_dir)
      ],
      file_outputs={}
      )
    name=name

class TAOExportClassificationOp(dsl.ContainerOp):
  def __init__(self, name, tao_mount_dir, api_key, input_file, output_file):
    super(TAOExportClassificationOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['classification'],
      arguments=[
        'export',
        '-m',
        '%s/%s' % (tao_mount_dir , input_file),
        '-k', api_key,
        '-o', '%s/%s' % (tao_mount_dir, output_file)
      ],
      file_outputs={}
      )
    name=name

# Added a version of Export to support INT8 creation for sample classification workflow

class TAOExportClassificationAdvancedOp(dsl.ContainerOp):
  def __init__(self, name, tao_mount_dir, api_key, input_file, output_file, calibration_file, data_type, num_batches, calibration_cache_file):
    super(TAOExportClassificationAdvancedOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['classification'],
      arguments=[
        'export',
        '-m', '%s/%s' % (tao_mount_dir , input_file),
        '-k', api_key,
        '-o', '%s/%s' % (tao_mount_dir, output_file),
        '--cal_data_file', '%s/%s' % (tao_mount_dir, calibration_file),
        '--data_type', data_type,
        '--batches', num_batches,
        '--cal_cache_file', '%s/%s' % (tao_mount_dir, calibration_cache_file),
        '-v'
      ],
      file_outputs={}
      )
    name=name

class TAOPruneOp(dsl.ContainerOp):
  def __init__(self, name, tao_mount_dir, api_key, pretrained_model, output_dir,
               pruning_threshold, equalization_criterion):
    super(TAOPruneOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['classification'],
      arguments=[
        'prune',
        '-k', api_key,
        '-m', '%s/%s' % (tao_mount_dir, pretrained_model),
        '-o', '%s/%s' % (tao_mount_dir, output_dir),
        '-pth', pruning_threshold,
        '-eq', equalization_criterion
      ],
      file_outputs={}
      )
    name=name

class TAOEvaluateClassificationOp(dsl.ContainerOp):
  def __init__(self, name, tao_mount_dir, api_key, spec_file):
    super(TAOEvaluateClassificationOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['classification'],
      arguments=[
        "evaluate",
        '-e', '%s/%s' % (tao_mount_dir, spec_file),
        '-k', api_key
      ],
      file_outputs={}
      )
    name=name

class TAOInferenceClassificationOp(dsl.ContainerOp):
  def __init__(self, name, tao_mount_dir, api_key, model, image_dir,
               batch_size, classmap, spec_file):
    super(TAOInferenceClassificationOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['classification'],
      arguments=[
        "inference",
        '-k', api_key,
        '-e', '%s/%s' % (tao_mount_dir, spec_file),
        '-m', '%s/%s' % (tao_mount_dir, model),
        '-d', '%s/%s' % (tao_mount_dir, image_dir),
        '-b', batch_size,
        '-cm', '%s/%s' % (tao_mount_dir, classmap)
      ],
      file_outputs={}
      )
    name=name

class TAOCalibrateClassificationOp(dsl.ContainerOp):
  def __init__(self, name, tao_mount_dir, spec_file, max_batches, calibration_file):
    super(TAOCalibrateClassificationOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['classification'],
      arguments=[
        'calibration_tensorfile',
        '-e', '%s/%s' % (tao_mount_dir, spec_file),
        '-m', max_batches,
        '-o', '%s/%s' % (tao_mount_dir, calibration_file),
      ],
      file_outputs={}
      )
    name=name

#
# Object Detection Routines
#
# Note: Not updated

class TAOTrainDetectionOp(dsl.ContainerOp):
  def __init__(self, name, tao_mount_dir, api_key, results_dir, spec_file, num_gpus, model_name):
    super(TAOTrainDetectionOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['tao-train'],
      arguments=[
        "detection",
        '--gpus', num_gpus,
        '-k', api_key,
        '-e', '%s/%s' % (tao_mount_dir, spec_file),
        '-r', '%s/%s' % (tao_mount_dir, results_dir),
        '-n', model_name
      ],
      file_outputs={}
      )
    name=name

class TAOEvaluateDetectionOp(dsl.ContainerOp):
  def __init__(self, name, tao_mount_dir, api_key, experiment_spec_file, model_file):
    super(TAOEvaluateDetectionOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['tao-evaluate'],
      arguments=[
        "detection",
        '-e', '%s/%s' % (tao_mount_dir, experiment_spec_file),
        '-k', api_key,
        '-m', '%s/%s' % (tao_mount_dir, model_file)
      ],
      file_outputs={}
      )
    name=name

class TAOInferenceDetectionOp(dsl.ContainerOp):
  def __init__(self, name, tao_mount_dir, api_key, model, inference_input,
               inference_output, batch_size, box_overlay, cluster_params_file,
               line_width, gpu_set, output_cov, output_bbox, kitti_dump):
    super(TAOInferenceDetectionOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['tao-infer'],
      arguments=[
        "detection",
        '-ek', encryption_key,
        '-m', '%s/%s' % (tao_mount_dir, model),
        '-i', '%s/%s' % (tao_mount_dir, inference_input),
        '-o', '%s/%s' % (tao_mount_dir, inference_output),
        '-bs', batch_size,
        '-k', kitti_dump,
        '-bo', box_overlay,
        '-cp', '%s/%s' % (tao_mount_dir, cluster_params_file),
        'lw', line_width,
        '-g', gpu_set,
        '--output_cov', output_cov,
        '--output_bbox', output_bbox,
      ],
      file_outputs={}
      )
    name=name

#
# TRT Operators
#

class TAOConvertTRTOp(dsl.ContainerOp):
  def __init__(self, name, tao_mount_dir, api_key, model, output_trt_file, output_layer, dims, input_type, max_trt_batch_size, precision, batch_size):
    super(TAOConvertTRTOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['converter'],
      arguments=[
        '%s/%s' % (tao_mount_dir, model),
        '-k', api_key,
        '-o', output_layer,
        '-d', dims,
        '-i', input_type,
        '-m', max_trt_batch_size,
        '-t', precision,
        '-e', '%s/%s' % (tao_mount_dir, output_trt_file),
        '-b', batch_size
      ],
      file_outputs={}
      )
    name=name

class TAOConvertTRTCalibrateOp(dsl.ContainerOp):
  def __init__(self, name, tao_mount_dir, api_key, model, calib_cache_file, output_trt_file, output_layer, dims, input_type, max_trt_batch_size, precision, batch_size):
    super(TAOConvertTRTCalibrateOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['converter'],
      arguments=[
        '%s/%s' % (tao_mount_dir, model),
        '-k', api_key,
        '-o', output_layer,
        '-c', '%s/%s' % (tao_mount_dir, calib_cache_file),
        '-d', dims,
        '-i', input_type,
        '-m', max_trt_batch_size,
        '-t', precision,
        '-e', '%s/%s' % (tao_mount_dir, output_trt_file),
        '-b', batch_size
      ],
      file_outputs={}
      )
    name=name

parse_key_values={"mean":"1", "median":"2", "max":"3", "min":"4", "90percent":"5"}
class TAOParseResultsOp(dsl.ContainerOp):
  def __init__(self, name, value_id, path_prefix, filename):
    super(TAOParseResultsOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['/bin/bash'],
      arguments=[
        '-c', "awk '{for(i=1;i<=NF;i++)if($i~/^-?[0-9]+\.[0-9]+$/){print $i}}' %s/%s | sed -n %sp > %s/%s.txt" % (path_prefix, filename, parse_key_values[value_id], path_prefix, value_id)
      ],
      file_outputs={'output': '%s/%s.txt' % (path_prefix, value_id)})
    name=name

class TAODeployOp(dsl.ContainerOp):
  def __init__(self, name, tao_mount_dir, model_file, destination):
    super(TAODeployOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['cp'],
      arguments=[
        '-r', "%s/%s %s" % (tao_mount_dir, model_file, destination)
      ],
      file_outputs={})
    name=name

class KubeflowFetchOp(dsl.ContainerOp):
  def __init__(self, path_prefix, filename):
    super(KubeflowFetchOp, self).__init__(
      name='FetchOp',
      image=__TAO_CONTAINER_VERSION__,
      command=['ls'],
      arguments=[
        '-lah', path_prefix
      ],
      file_outputs={'output': '%s/%s' % (path_prefix, filename)})

class KubeflowLSOp(dsl.ContainerOp):
  def __init__(self, name, path):
    super(KubeflowLSOp, self).__init__(
      name=name,
      image=__TAO_CONTAINER_VERSION__,
      command=['/bin/bash'],
      arguments=[
            '-c', 'ls', '-lah', path
        ],
      file_outputs={}
      )
    name=name

class FlipCoinOp(dsl.ContainerOp):
  def __init__(self, name):
    super(FlipCoinOp, self).__init__(
        name=name,
        image='python:alpine3.6',
        command=['sh', '-c'],
        arguments=['python -c "import random; result = \'heads\' if random.randint(0,1) == 0 '
                   'else \'tails\'; print(result)" | tee /tmp/output'],
        file_outputs={'output': '/tmp/output'})

class PrintOp(dsl.ContainerOp):
  def __init__(self, name, command):
    super(PrintOp, self).__init__(
        name=name,
        image='alpine:3.6',
        command=command)
