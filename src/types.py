from typing import List, Literal, TypedDict, NotRequired

from torch import Tensor

SampleID = int
TextContent = str


# Terrible Python Type Hints Made Us Do This
class DataIncludeImage(TypedDict):
    image: List[Tensor]
    text: NotRequired[List[TextContent]]


class DataIncludeText(TypedDict):
    image: NotRequired[List[Tensor]]
    text: List[TextContent]


class DataIncludeImageAndText(TypedDict):
    image: List[Tensor]
    text: List[TextContent]


TaskName = Literal[
    'input_of_query',
    'sentiment_analysis',
    'machine_translation',
    'image_classification',
    'object_detection',
    'image_super_resolution',
    'image_colorization',
    'image_denoising',
    'image_deblurring',
    'image_captioning',
    'text_to_image',
    'question_answering',
    'visual_question_answering',
    'text_summarization',
    'text_generation',
    'mask_filling'
]
ModelName = str


class CostInfo(TypedDict):
    """
    CostInfo is a dictionary that contains the cost information of a tool.
    """
    exec_time: float
    short_term_cpu_memory: float
    short_term_gpu_memory: float
