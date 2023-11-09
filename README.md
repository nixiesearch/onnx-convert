# An ONNX conversion script for embedding models

This project is based on an ONNX conversion script taken from [xenova/transformers.js](https://github.com/xenova/transformers.js/tree/main/scripts) project.

## Usage

```shell
pip install -r requirements.txt

python convert.py --help
```

Command-line parameters are:

```
usage: convert.py [-h] --model_id MODEL_ID [--quantize [QUANTIZE]] [--output_parent_dir OUTPUT_PARENT_DIR] [--task TASK] [--opset OPSET] [--device DEVICE] [--skip_validation [SKIP_VALIDATION]]
                  [--per_channel [PER_CHANNEL]] [--no_per_channel] [--reduce_range [REDUCE_RANGE]] [--no_reduce_range] [--weight_type WEIGHT_TYPE] [--optimizer_level OPTIMIZER_LEVEL]

options:
  -h, --help            show this help message and exit
  --model_id MODEL_ID   Model identifier (default: None)
  --quantize [QUANTIZE]
                        Whether to quantize the model. (default: False)
  --output_parent_dir OUTPUT_PARENT_DIR
                        Path where the converted model will be saved to. (default: ./models/)
  --task TASK           The task to export the model for. If not specified, the task will be auto-inferred based on the model. (default: sentence-similarity)
  --opset OPSET         If specified, ONNX opset version to export the model with. Otherwise, the default opset will be used. (default: None)
  --device DEVICE       The device to use to do the export. (default: cpu)
  --skip_validation [SKIP_VALIDATION]
                        Whether to skip validation of the converted model (default: False)
  --per_channel [PER_CHANNEL]
                        Whether to quantize weights per channel (default: True)
  --no_per_channel      Whether to quantize weights per channel (default: False)
  --reduce_range [REDUCE_RANGE]
                        Whether to quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode (default: True)
  --no_reduce_range     Whether to quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode (default: False)
  --weight_type WEIGHT_TYPE
                        Which underlying integer format should be used. Options are QInt8/QUInt8 (default: QUInt8)
  --optimizer_level OPTIMIZER_LEVEL
                        ONNX optimizer level. Options are [0, 1, 2, 99] (default: 1)
```

## Differences with the xenova/transformers.js

This script was extended with the following features:

* support for ONNX transformer optimization pass
* selection of QUint8/QInt8 storage

## License

Apache 2.0