from typing import Dict, List, Type

from src.config import MODEL_REGISTRY, log
from src.types import TaskName, ModelName
from .grouped_tools import (
    Tool,
    GroupedTools,
    SentimentAnalysisTools,
    MachineTranslationTools,
    ImageClassificationTools,
    ObjectDetectionTools,
    ImageSuperResolutionTools,
    ImageColorizationTools,
    ImageDenoisingTools,
    ImageDeblurringTools,
    ImageCaptioningTools,
    TextToImageTools,
    QuestionAnsweringTools,
    VisualQuestionAnsweringTools,
    TextSummarizationTools,
    TextGenerationTools,
    MaskFillingTools
)


class ToolManager:
    """
    Manages different task-specific tool groups and their associated models.

    Attributes:
        tool_cls_groups: A mapping from a task name to its corresponding tool group class.
        tool_groups: An instance-level dictionary that stores each task's tool group once loaded.
    """
    tool_cls_groups: Dict[TaskName, Type[GroupedTools]] = {
        'sentiment_analysis': SentimentAnalysisTools,
        'machine_translation': MachineTranslationTools,
        'image_classification': ImageClassificationTools,
        'object_detection': ObjectDetectionTools,
        'image_super_resolution': ImageSuperResolutionTools,
        'image_colorization': ImageColorizationTools,
        'image_denoising': ImageDenoisingTools,
        'image_deblurring': ImageDeblurringTools,
        'image_captioning': ImageCaptioningTools,
        'text_to_image': TextToImageTools,
        'question_answering': QuestionAnsweringTools,
        'visual_question_answering': VisualQuestionAnsweringTools,
        'text_summarization': TextSummarizationTools,
        'text_generation': TextGenerationTools,
        'mask_filling': MaskFillingTools,
    }

    def __init__(self):
        self.tool_groups: Dict[TaskName, GroupedTools] = {}

    def load_model(self, task_name: TaskName, model_name: ModelName) -> None:
        """
        Load a specific model into the corresponding task's tool group.

        Args:
            task_name: The name of the task (e.g., 'sentiment_analysis').
            model_name: The name of the model to load for this task.
        """
        if task_name not in self.tool_groups:
            # Instantiate the tool group class if it doesn't exist yet.
            model_cls = self.tool_cls_groups[task_name]
            self.tool_groups[task_name] = model_cls()
        self.tool_groups[task_name].load_model(model_name)

    def load_models(self, task_name: TaskName = 'all_tasks', model_name: ModelName = 'all_models') -> None:
        """
        Load one or more models based on the provided task_name and model_name.

        Args:
            task_name: The task to load (default 'all_tasks' means load for all tasks).
            model_name: The model to load (default 'all_models' means load all models).

        Raises:
            ValueError: If 'all_tasks' is specified along with a specific model_name.
        """
        if task_name == 'all_tasks':
            # Load all tasks and all models
            if model_name == 'all_models':
                for task_name_item in MODEL_REGISTRY:
                    log.opt(colors=True).info(f'Initializing all models for task: <blue>{task_name_item}</blue>')
                    for model_name_item in MODEL_REGISTRY[task_name_item]:
                        self.load_model(task_name_item, model_name_item)
            else:
                raise ValueError(
                    'Invalid parameters: task_name is "all_tasks" but a specific model_name is provided.'
                )
        else:
            # Load models for a specific task
            if model_name == 'all_models':
                for model_name_item in MODEL_REGISTRY[task_name]:
                    self.load_model(task_name, model_name_item)
            else:
                self.load_model(task_name, model_name)

    def list_models(self) -> Dict[TaskName, List[ModelName]]:
        """
        List the currently loaded models for each task.

        Returns:
            A dictionary mapping each task to the list of loaded model names.
        """
        # Aggregate the results from each task's tool group.
        result: Dict[TaskName, List[ModelName]] = {}
        for task_name, group_tool in self.tool_groups.items():
            # Each group's list_models() is assumed to return {task_name: [model_list]} or similar.
            result.update(group_tool.list_models())
        return result

    def get_model(self, task_name: TaskName, model_name: ModelName) -> Tool:
        """
        Retrieve a loaded model from a specific task group.
        If model_name is None, the default model is used (the first in MODEL_REGISTRY).

        Args:
            task_name: The name of the task to retrieve from.
            model_name: The specific model name to retrieve, or None to use the default.

        Returns:
            The tool instance (model) requested.

        Raises:
            ValueError: If no models are available for the given task.
        """
        # If no model name is provided, use the default (the first in the registry).
        if model_name is None:
            available_models = list(MODEL_REGISTRY[task_name].keys())
            if not available_models:
                raise ValueError(f"No available models for task '{task_name}'.")
            log.opt(colors=True).info(
                f'No model_name specified in {task_name}, using default model: {available_models[0]}')
            model_name = available_models[0]

        # Retrieve or create the group for this task
        group = self.tool_groups.get(task_name)
        if group is None:
            model_cls = self.tool_cls_groups[task_name]
            group = model_cls()
            self.tool_groups[task_name] = group

        # Return the requested model (tool) instance
        return group.get_model(model_name)


tool_manager: ToolManager
_tool_manager = None


def __getattr__(name):
    """
    Custom attribute access for the ToolManager instance.
    This allows for lazy initialization of the ToolManager.

    Args:
        name: The name of the attribute being accessed.

    Returns:
        The ToolManager instance or the requested attribute.
    """
    if name == 'tool_manager':
        global _tool_manager
        if _tool_manager is None:
            _tool_manager = ToolManager()
        return _tool_manager

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
