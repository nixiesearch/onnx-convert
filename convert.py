
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Optional, Set
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser
)

import onnx
from optimum.exporters.onnx import main_export, export_models
from optimum.exporters.tasks import TasksManager
from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType,
    QuantFormat
)

from onnxruntime.transformers import optimizer, float16

# Based on xenova/transformers.js translation script.

@dataclass
class ConversionArguments:
    """
    Arguments used for converting HuggingFace models to onnx.
    """

    model_id: str = field(
        metadata={
            "help": "Model identifier"
        }
    )
    quantize: Optional[str] = field(
        default=None,
        metadata={
            "help": "To which format to quantize the model. Options are QInt8/QUInt8/Float16"
        }
    )
    output_parent_dir: str = field(
        default='./models/',
        metadata={
            "help": "Path where the converted model will be saved to."
        }
    )

    task: Optional[str] = field(
        default='sentence-similarity',
        metadata={
            "help": (
                "The task to export the model for. If not specified, the task will be auto-inferred based on the model."
            )
        }
    )

    opset: int = field(
        default=None,
        metadata={
            "help": (
                "If specified, ONNX opset version to export the model with. Otherwise, the default opset will be used."
            )
        }
    )

    device: str = field(
        default='cpu',
        metadata={
            "help": 'The device to use to do the export.'
        }
    )
    skip_validation: bool = field(
        default=False,
        metadata={
            "help": "Whether to skip validation of the converted model"
        }
    )

    per_channel: bool = field(
        default=True,
        metadata={
            "help": "Whether to quantize weights per channel"
        }
    )
    reduce_range: bool = field(
        default=True,
        metadata={
            "help": "Whether to quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode"
        }
    )

    optimize: int = field(
        default=1,
        metadata={
            "help": "ONNX optimizer level. Options are [0, 1, 2, 99]"
        }
    )

def get_operators(model: onnx.ModelProto) -> Set[str]:
    operators = set()

    def traverse_graph(graph):
        for node in graph.node:
            operators.add(node.op_type)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    subgraph = attr.g
                    traverse_graph(subgraph)

    traverse_graph(model.graph)
    return operators


def quantize(model_names_or_paths, conv_args, **quantize_kwargs):
    """
    Quantize the weights of the model from float32 to int8 to allow very efficient inference on modern CPU

    Args:
        onnx_model_path: Path to location the exported ONNX model is stored
    Returns: The Path generated for the quantized
    """

    quantize_config = dict(
        **quantize_kwargs,
        per_model_config={}
    )

    

    for model in model_names_or_paths:
        print(f'Processing model file {model}')
        directory_path = os.path.dirname(model)
        file_name_without_extension = os.path.splitext(
            os.path.basename(model))[0]

        loaded_model = onnx.load_model(model)
        print('ONNX model loaded')
        op_types = get_operators(loaded_model)

        if conv_args.optimize > 0:
            print(f'Optimizing model with level={conv_args.optimize}')
            optimized_model = optimizer.optimize_model(model, model_type='bert', opt_level=conv_args.optimize)
            optimized_model_file = f'{file_name_without_extension}_opt{conv_args.optimize}_Float32.onnx'
            optimized_model.save_model_to_file(os.path.join(directory_path, optimized_model_file))

            
            if conv_args.quantize:
                print(f'Optimization done, quantizing to {conv_args.quantize}')
                quantized_file = f'{file_name_without_extension}_opt{conv_args.optimize}_{conv_args.quantize}.onnx'
                quantize_impl(conv_args, directory_path, optimized_model_file, quantized_file, **quantize_kwargs)
            else:
                print('Skipping quantization')
        else:
            if conv_args.quantize:
                print(f'No optimization enabled, quantizing to {conv_args.quantize}')
                quantized_file = f'{file_name_without_extension}_opt0_{conv_args.quantize}.onnx'
                quantize_impl(conv_args, directory_path, model, quantized_file, **quantize_kwargs)
            else:
                print('Skipping quantization')

def quantize_impl(conv_args, directory_path, model_file, out_file, **quantize_kwargs):
    if conv_args.quantize in ["QUInt8", "QInt8"]:    
        wt = QuantType.QUInt8

        if conv_args.quantize == "QUInt8":
            wt = QuantType.QUInt8
        elif conv_args.quantize == "QInt8":
            wt = QuantType.QInt8


        quantize_dynamic(
                        model_input=os.path.join(directory_path, model_file),
                        model_output=os.path.join(directory_path, out_file),
                        weight_type=wt,
                        extra_options=dict(
                            EnableSubgraph=True
                        ),
                        **quantize_kwargs
                    )
    elif conv_args.quantize == "Float16":
        model = onnx.load(os.path.join(directory_path, model_file))
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, os.path.join(directory_path, out_file))
    


def main():
    
    parser = HfArgumentParser((ConversionArguments, ))
    conv_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    print(f'Conversion config: {conv_args}')
    model_id = conv_args.model_id

    output_model_folder = os.path.join(conv_args.output_parent_dir, model_id)

    # Create output folder
    os.makedirs(output_model_folder, exist_ok=True)

    export_kwargs = dict(
        model_name_or_path=model_id,
        output=output_model_folder,
        task=conv_args.task,
        opset=conv_args.opset,
        device=conv_args.device,
        do_validation=not conv_args.skip_validation,
    )
    print('Exporting model to ONNX')
    main_export(**export_kwargs)
    
    print('Export done')

    quantize_config = {'per_channel': conv_args.per_channel, 'reduce_range': conv_args.reduce_range}

    quantize([
        os.path.join(output_model_folder, x)
        for x in os.listdir(output_model_folder)
        if x == 'model.onnx'
    ], conv_args, **quantize_config)

    # Step 3. Move .onnx files to the 'onnx' subfolder
    os.makedirs(os.path.join(output_model_folder, 'onnx'), exist_ok=True)
    for file in os.listdir(output_model_folder):
        if file.endswith(('.onnx', '.onnx_data')):
            shutil.move(os.path.join(output_model_folder, file),
                        os.path.join(output_model_folder, 'onnx', file))


if __name__ == '__main__':
    main()
