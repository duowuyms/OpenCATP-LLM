"""
Claim: Some tool cost-related setups in this file may vary on different hardware devices.
For instance, in GlobalMetricConfig, we provide the information of memory consumption of tools.
This information was measured on our GPU server with two 32GB V100.
This information may be different on other GPU servers.
"""
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Literal, Optional
from loguru import logger as log
from .types import TaskName, ModelName

current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
home_dir = os.path.dirname(src_dir)

# Constants that should typically be reset based on the device environment
PRETRAINED_LLM_DIR = "/home/data/pretrained_llms/"

TOOL_DEVICE_LIST = ["cuda:0"]
EVALUATOR_DEVICE_LIST = ["cuda:1"]
# the minimum GPU memory allocation limit, default is 3GB
TOOL_GPU_MEMORY_ALLOC_LIMIT = 3 * 1024 ** 3

# log config when using loguru
LOG_FORMAT_CONSOLE = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
    " | <level>{level:<7}</level>"
    " | <cyan>{name}</cyan>:<cyan>{line}</cyan>"
    " - <level>{message}</level>"
)

DEFAULT_START_TASK_NAME: TaskName = "input_of_query"


@dataclass
class GlobalPathConfig:
    # recommendation: make hf_cache, data_path, finetune_path as a soft link.
    hf_cache = os.path.join(home_dir, "hf_cache/")
    model_cache = os.path.join(home_dir, "model_cache/")
    data_path = os.path.join(home_dir, "dataset/")
    result_path = os.path.join(home_dir, "results/")
    finetune_path = os.path.join(home_dir, 'finetune_models')


@dataclass
class GlobalTaskConfig:
    default_train_seq_tasks = [
        1, 2, 3, 4, 5, 7, 9, 10, 11, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 32, 33, 34, 35, 37, 38, 39, 41, 42, 43, 44, 45,
        47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 63, 64, 65, 66,
        67, 68, 70, 71, 72, 73, 75, 76, 77, 79, 80, 82, 83, 84, 85, 86,
    ]
    default_test_seq_tasks = [
        0, 6, 8, 12, 13, 20, 21, 31, 36, 40, 46, 51, 61, 62, 69, 74, 78, 81,
    ]
    default_train_nonseq_tasks = [
        201, 206, 208, 210, 211, 213, 214, 216, 217, 220, 222, 223,
    ]
    default_test_nonseq_tasks = [
        200, 202, 203, 204, 205, 207, 209, 212, 215, 218, 219, 221,
    ]
    # default_test_task_samples = json.load(open(os.path.join(GlobalPathConfig.data_path, 'test_task_samples.json'), 'r'))


@dataclass
class GlobalMetricsConfig:
    MIN_SCORE = 0
    MAX_SCORE = 1
    MIN_COST = 0
    MAX_COST = 0.5669374317944147
    ALPHA = 0.5

    score_penalty = -2  # penalty assigned to the scores of invalid plans
    cost_penalty = 2  # penalty assigned to the costs of invalid plans

    tools_cpu_long_term_mem = {
        "image_classification": 1788.50390625,
        "image_colorization": 1626.96875,
        "object_detection": 1684.6875,
        "image_deblurring": 1693.83984375,
        "image_denoising": 1690.15625,
        "image_super_resolution": 1540.8515625,
        "image_captioning": 2449.8359375,
        "text_to_image": 6746.109375,  # not used in the current OpenCATP, leave for possible use in the future
        "visual_question_answering": 1953.0234375,
        # not used in the current OpenCATP, leave for possible use in the future
        "sentiment_analysis": 1719.6875,
        "question_answering": 1696.2734375,  # not used in the current OpenCATP, leave for possible use in the future
        "text_summarization": 3321.2578125,
        "text_generation": 1937.03125,  # not used in the current OpenCATP, leave for possible use in the future
        "machine_translation": 2388.72265625,
        "mask_filling": 1712.41796875,  # not used in the current OpenCATP, leave for possible use in the future
    }
    tools_gpu_long_term_mem = {
        "image_classification": 330.2294921875,
        "image_colorization": 131.31689453125,
        "object_detection": 234.7177734375,
        "image_deblurring": 99.74462890625,
        "image_denoising": 99.67431640625,
        "image_super_resolution": 47.52490234375,
        "image_captioning": 937.234375,
        "text_to_image": 5252.0234375,
        "visual_question_answering": 449.14599609375,
        "sentiment_analysis": 256.49755859375,
        "question_answering": 249.19384765625,
        "text_summarization": 1550.06689453125,
        "text_generation": 487.46875,
        "machine_translation": 850.3095703125,
        "mask_filling": 256.61376953125,
    }

    # Long-term cpu memory pricing tiers. Data format: {memory_size(MB): price(USD)} per ms
    cpu_long_memory_pricing = {
        128: 0.0000000021,
        512: 0.0000000083,
        1024: 0.0000000167,
        1536: 0.0000000250,
        2048: 0.0000000333,
        3072: 0.0000000500,
        4096: 0.0000000667,
        5120: 0.0000000833,
        6144: 0.0000001000,
        7168: 0.0000001167,
        8192: 0.0000001333,
        9216: 0.0000001500,
        10240: 0.0000001667,
    }
    # Short-term cpu memory pricing per MB.
    cpu_short_memory_pricing_per_mb = 0.0000000000000302
    # Note that the AWS Lambda does not provide pricing strategy for GPU resources.
    # We use set the GPU prices as three times of CPU prices, according to the following article:
    # https://news.rice.edu/news/2021/rice-intel-optimize-ai-training-commodity-hardware
    gpu_long_memory_pricing = {k: v * 3 for k, v in cpu_long_memory_pricing.items()}
    gpu_short_memory_pricing_per_mb = cpu_short_memory_pricing_per_mb * 3
    price_per_request = 0.0000002


@dataclass
class GlobalToolConfig:
    tool_token_start = 80000
    sop_token = tool_token_start
    eop_token = sop_token + 1

    dependency_token_start = 90000
    sod_token = dependency_token_start
    eod_token = sod_token + 1

    max_num_tokens = 50
    max_num_generated_tokens = 40
    max_ep_len = 100  # max length of episode (= max number of tokens to be generated)

    # tool-token mapping
    tool_token_vocabulary = {
        "image_classification": sop_token + 2,
        "image_colorization": sop_token + 3,
        "object_detection": sop_token + 4,
        "image_deblurring": sop_token + 5,
        "image_denoising": sop_token + 6,
        "image_super_resolution": sop_token + 7,
        "image_captioning": sop_token + 8,
        "text_to_image": sop_token + 9,
        "visual_question_answering": sop_token + 10,
        "sentiment_analysis": sop_token + 11,
        "question_answering": sop_token + 12,
        "text_summarization": sop_token + 13,
        "text_generation": sop_token + 14,
        "machine_translation": sop_token + 15,
        "mask_filling": sop_token + 16,
    }
    tool_token_vocabulary_reverse = {v: k for k, v in tool_token_vocabulary.items()}

    # dependency-token mapping
    dependency_token_vocabulary = {
        "image_classification": sod_token + 2,
        "image_colorization": sod_token + 3,
        "object_detection": sod_token + 4,
        "image_deblurring": sod_token + 5,
        "image_denoising": sod_token + 6,
        "image_super_resolution": sod_token + 7,
        "image_captioning": sod_token + 8,
        "text_to_image": sod_token + 9,
        "visual_question_answering": sod_token + 10,
        "sentiment_analysis": sod_token + 11,
        "question_answering": sod_token + 12,
        "text_summarization": sod_token + 13,
        "text_generation": sod_token + 14,
        "machine_translation": sod_token + 15,
        "mask_filling": sod_token + 16,
        DEFAULT_START_TASK_NAME: sod_token + 17,
    }
    dependency_token_vocabulary_reverse = {
        v: k for k, v in dependency_token_vocabulary.items()
    }

    tool_io_dict = {
        "image_colorization": ["image", "image"],
        "image_denoising": ["image", "image"],
        "image_deblurring": ["image", "image"],
        "image_super_resolution": ["image", "image"],
        "image_classification": ["image", "text"],
        "image_captioning": ["image", "text"],
        "object_detection": ["image", "text"],
        "text_summarization": ["text", "text"],
        "text_generation": ["text", "text"],
        "machine_translation": ["text", "text"],
        "mask_filling": ["text", "text"],
        "sentiment_analysis": ["text", "text"],
        "text_to_image": ["text", "image"],
        "question_answering": ["text-text", "text"],
        "visual_question_answering": ["image-text", "text"],
    }

    tool_token_io_dict = {
        tool_token_vocabulary["image_colorization"]: ["image", "image"],
        tool_token_vocabulary["image_denoising"]: ["image", "image"],
        tool_token_vocabulary["image_deblurring"]: ["image", "image"],
        tool_token_vocabulary["image_super_resolution"]: ["image", "image"],
        tool_token_vocabulary["image_classification"]: ["image", "text"],
        tool_token_vocabulary["image_captioning"]: ["image", "text"],
        tool_token_vocabulary["object_detection"]: ["image", "text"],
        tool_token_vocabulary["text_summarization"]: ["text", "text"],
        tool_token_vocabulary["text_generation"]: ["text", "text"],
        tool_token_vocabulary["machine_translation"]: ["text", "text"],
        tool_token_vocabulary["mask_filling"]: ["text", "text"],
        tool_token_vocabulary["sentiment_analysis"]: ["text", "text"],
        tool_token_vocabulary["text_to_image"]: ["text", "image"],
        tool_token_vocabulary["question_answering"]: ["text-text", "text"],
        tool_token_vocabulary["visual_question_answering"]: ["image-text", "text"],
    }

    tool_io_dict_collection = {
        "in:image-out:image": [
            "image_colorization",
            "image_denoising",
            "image_deblurring",
            "image_super_resolution",
        ],
        "in:image-out:text": [
            "image_classification",
            "image_captioning",
            "object_detection",
        ],
        "in:text-out:text": [
            "text_summarization",
            "text_generation",
            "machine_translation",
            "mask_filling",
            "sentiment_analysis",
        ],
        "in:text-out:image": ["text_to_image"],
        "in:image,text-out:text": ["visual_question_answering"],
        "in:text,text-out:text": ["question_answering"],
    }
    # same as above, but using tokens
    tool_token_io_dict_collection = {
        "in:image-out:image": [
            tool_token_vocabulary["image_colorization"],
            tool_token_vocabulary["image_denoising"],
            tool_token_vocabulary["image_deblurring"],
            tool_token_vocabulary["image_super_resolution"],
        ],
        "in:image-out:text": [
            tool_token_vocabulary["image_classification"],
            tool_token_vocabulary["image_captioning"],
            tool_token_vocabulary["object_detection"],
        ],
        "in:text-out:text": [
            tool_token_vocabulary["text_summarization"],
            tool_token_vocabulary["text_generation"],
            tool_token_vocabulary["machine_translation"],
            tool_token_vocabulary["mask_filling"],
            tool_token_vocabulary["sentiment_analysis"],
        ],
        "in:text-out:image": [tool_token_vocabulary["text_to_image"]],
        "in:image,text-out:text": [tool_token_vocabulary["visual_question_answering"]],
        "in:text,text-out:text": [tool_token_vocabulary["question_answering"]],
    }

    task_io_dict = {
        "in:image-out:image": set(range(0, 15)),
        "in:image-out:text": set(range(15, 105)),
        "in:text-out:image": set(range(105, 108)),
        "in:text-out:text": set(range(108, 126)),
        "in:image,text-out:text": set(range(126, 171)),
        "in:text,text-out:text": set(range(171, 188)),
    }

    tool_dependencies = {
        # e.g. 'image_colorization': ['image_super_resolution', ...] means that image_colorization depends on image_super_resolution
        "image_colorization": [
            "image_super_resolution",
            "image_deblurring",
            "image_denoising",
        ],
        "image_super_resolution": [
            "image_colorization",
            "image_deblurring",
            "image_denoising",
        ],
        "image_deblurring": [
            "image_colorization",
            "image_super_resolution",
            "image_denoising",
        ],
        "image_denoising": [
            "image_colorization",
            "image_super_resolution",
            "image_deblurring",
        ],
        "image_captioning": [],
        "image_classification": [],
        "object_detection": [],
        "machine_translation": [],
        "sentiment_analysis": [],
        "text_summarization": [],
        "mask_filling": [],
        "text_generation": [],
        "text_to_image": [],
    }
    tool_dependencies_reverse = {
        "image_colorization": [
            "image_super_resolution",
            "image_deblurring",
            "image_denoising",
        ],
        "image_super_resolution": [
            "image_colorization",
            "image_deblurring",
            "image_denoising",
        ],
        "image_deblurring": [
            "image_colorization",
            "image_super_resolution",
            "image_denoising",
        ],
        "image_denoising": [
            "image_colorization",
            "image_super_resolution",
            "image_deblurring",
        ],
        "image_captioning": [],
        "image_classification": [],
        "object_detection": [],
        "machine_translation": [],
        "sentiment_analysis": [],
        "text_summarization": [],
        "mask_filling": [],
        "text_generation": [],
        "text_to_image": [],
    }

    tool_prices = {
        'image_colorization': [0.02064448408770396, 0.030365631874341062, 0.03283418516591483, 0.20662706942016817],
        'mask_filling': [0.0007191801038195274, 0.0011161143430851086, 0.0011966995137895854, 0.001151140264732181],
        'image_captioning': [0.07681041316118693, 0.0789306347162341, 0.0856651295420851, 0.15344758910825465],
        'image_classification': [0.0018386416371498659, 0.002039574896674113, 0.001632744822041972,
                                 0.003468827901294199],
        'image_deblurring': [0.044596890616843535, 0.0763150490990877, 0.1770620811739597, 1.1910876856898132],
        'image_denoising': [0.03310938232453873, 0.05553857351136441, 0.126780926718445, 0.8438447906436202],
        'image_super_resolution': [0.12710071314082524, 0.17339502868541215, 0.3204404321613257, 1.8004475979612815],
        'machine_translation': [0.0514501296757946, 0.07495833137293198, 0.17899568650893222, 0.1265572219738673],
        'object_detection': [0.008221186114819342, 0.007555937255849043, 0.007457512004008459, 0.010285611128207868],
        'question_answering': [0.000651189057222963, 0.0005230975314769352, 0.0005303616164152156,
                               0.000673727151311317],
        'sentiment_analysis': [0.0009301707022420854, 0.0007762505455430947, 0.0007021147041751504,
                               0.0009259290352010135],
        'text_generation': [0.01703311232198894, 0.002373261885790527, 0.003585298728325665, 0.0037642009109413624],
        'text_summarization': [0.4399955272843922, 0.44162034375878384, 0.47984100235254135, 0.4836095130016943],
        'text_to_image': [30.41944787973723, 30.44826792797344, 30.441494159301133, 30.410528955212857],
        'visual_question_answering': [0.004567494635659777, 0.004190168623573798, 0.005081711239834369,
                                      0.009299321594933435]}


@dataclass
class GlobalDataConfig:
    image_sizes = [490 * 402, 582 * 578, 954 * 806, 1921 * 2624]
    text_lengths = [149, 2009, 4464, 7003]


@dataclass
class ModelConfig:
    task_name: Literal[
        "sentiment_analysis",
        "image_classification",
        "image_colorization",
        "object_detection",
        "image_super_resolution",
        "image_captioning",
        "text_to_image",
        "question_answering",
        "text_summarization",
        "text_generation",
        "visual_question_answering",
        "machine_translation",
        "mask_filling",
        "image_deblurring",
        "image_denoising",
    ]
    model_name: str
    source: Literal["huggingface", "github", "others"]
    hf_url: Optional[str]


MODEL_REGISTRY: Dict[TaskName, Dict[ModelName, ModelConfig]] = {
    "sentiment_analysis": {
        "distilbert-sst2": ModelConfig(
            task_name="sentiment_analysis",
            model_name="distilbert-sst2",
            source="huggingface",
            hf_url="distilbert-base-uncased-finetuned-sst-2-english",
        ),
        "bert-base-multilingual": ModelConfig(
            task_name="sentiment_analysis",
            model_name="bert-base-multilingual",
            source="huggingface",
            hf_url="nlptown/bert-base-multilingual-uncased-sentiment"
        ),
        "twitter-roberta-base": ModelConfig(
            task_name="sentiment_analysis",
            model_name="twitter-roberta-base",
            source="huggingface",
            hf_url="cardiffnlp/twitter-roberta-base-sentiment-latest",
        ),
    },
    "image_classification": {
        "vit-base": ModelConfig(
            task_name="image_classification",
            model_name="vit-base",
            source="huggingface",
            hf_url="google/vit-base-patch16-224",
        ),
        "resnet-50": ModelConfig(
            task_name="image_classification",
            model_name="resnet-50",
            source="huggingface",
            hf_url="microsoft/resnet-50",
        ),
        "mobilenet-v2": ModelConfig(
            task_name="image_classification",
            model_name="mobilenet-v2",
            source="huggingface",
            hf_url="google/mobilenet_v2_1.0_224",
        ),
    },
    "image_colorization": {
        "siggraph17": ModelConfig(
            task_name="image_colorization",
            model_name="siggraph17",
            source="github",
            hf_url=None,
        )
    },
    "object_detection": {
        "detr-resnet-101": ModelConfig(
            task_name="object_detection",
            model_name="detr-resnet-101",
            source="huggingface",
            hf_url="facebook/detr-resnet-101",
        ),
        "detr-resnet-50": ModelConfig(
            task_name="object_detection",
            model_name="detr-resnet-50",
            source="huggingface",
            hf_url="facebook/detr-resnet-50",
        ),
        "yolo-v8": ModelConfig(
            task_name="object_detection",
            model_name="yolo-v8",
            source="others",
            hf_url=None,
        ),
    },
    "image_super_resolution": {
        "swin2sr": ModelConfig(
            task_name="image_super_resolution",
            model_name="swin2sr",
            source="huggingface",
            hf_url="caidas/swin2SR-classical-sr-x2-64",
        )
    },
    "image_captioning": {
        "vit-gpt2": ModelConfig(
            task_name="image_captioning",
            model_name="vit-gpt2",
            source="huggingface",
            hf_url="nlpconnect/vit-gpt2-image-captioning",
        ),
        "blip-large": ModelConfig(
            task_name="image_captioning",
            model_name="blip-large",
            source="huggingface",
            hf_url="Salesforce/blip-image-captioning-large",
        ),
        "git-base-coco": ModelConfig(
            task_name="image_captioning",
            model_name="git-base-coco",
            source="huggingface",
            hf_url="microsoft/git-base-coco",
        )
    },
    # "text_to_image": {
    #     "stable-diffusion-v1-4": ModelConfig(
    #         task_name="text_to_image",
    #         model_name="stable-diffusion-v1-4",
    #         source="huggingface",
    #         hf_url="CompVis/stable-diffusion-v1-4",
    #     )
    # },
    # "question_answering": {
    #     "distilbert-squad": ModelConfig(
    #         task_name="question_answering",
    #         model_name="distilbert-squad",
    #         source="huggingface",
    #         hf_url="distilbert-base-cased-distilled-squad",
    #     )
    # },
    "text_summarization": {
        "bart-cnn": ModelConfig(
            task_name="text_summarization",
            model_name="bart-cnn",
            source="huggingface",
            hf_url="facebook/bart-large-cnn",
        )
    },
    "text_generation": {
        "gpt2-base": ModelConfig(
            task_name="text_generation",
            model_name="gpt2-base",
            source="huggingface",
            hf_url="gpt2",
        )
    },
    "visual_question_answering": {
        "vilt-vqa": ModelConfig(
            task_name="visual_question_answering",
            model_name="vilt-vqa",
            source="huggingface",
            hf_url="dandelin/vilt-b32-finetuned-vqa",
        )
    },
    "machine_translation": {
        "t5-base": ModelConfig(
            task_name="machine_translation",
            model_name="t5-base",
            source="huggingface",
            hf_url="t5-base",
        ),
        "t5-small": ModelConfig(
            task_name="machine_translation",
            model_name="t5-small",
            source="huggingface",
            hf_url="t5-small",
        ),
        "t5-large": ModelConfig(
            task_name="machine_translation",
            model_name="t5-large",
            source="huggingface",
            hf_url="t5-large",
        )
    },
    "mask_filling": {
        "distilbert-mlm": ModelConfig(
            task_name="mask_filling",
            model_name="distilbert-mlm",
            source="huggingface",
            hf_url="distilbert-base-uncased",
        )
    },
    "image_deblurring": {
        "restormer-deblur": ModelConfig(
            task_name="image_deblurring",
            model_name="restormer-deblur",
            source="github",
            hf_url=None,
        ),
        "MPRNet": ModelConfig(
            task_name="image_deblurring",
            model_name="MPRNet",
            source="github",
            hf_url=None,
        ),
    },
    "image_denoising": {
        "restormer-denoise": ModelConfig(
            task_name="image_denoising",
            model_name="restormer-denoise",
            source="github",
            hf_url=None,
        ),
        "MPRNet": ModelConfig(
            task_name="image_denoising",
            model_name="MPRNet",
            source="github",
            hf_url=None,
        ),
    },
}

log.remove()
log.add(sys.stdout, level="INFO", colorize=True, format=LOG_FORMAT_CONSOLE)
log.add(
    f'{current_file_path}/../../logs/{datetime.now().strftime("%Y-%m-%d")}.log',
    level="DEBUG",
)
