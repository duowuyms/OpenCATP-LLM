from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Callable
from runpy import run_path
import os
import requests

from skimage.util import img_as_ubyte
import torch.nn.functional as F
import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForImageTextToText,
    # AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForObjectDetection,
    ViTImageProcessor,
    ViTForImageClassification,
    Swin2SRForImageSuperResolution,
    ViltProcessor,
    ViltForQuestionAnswering,
    VisionEncoderDecoderModel,
)

from src.config import log, GlobalPathConfig, ModelConfig, MODEL_REGISTRY
from src.types import (
    TaskName,
    ModelName,
    DataIncludeImage,
    DataIncludeText,
    DataIncludeImageAndText,
)
from .tool import Tool


class GroupedTools(ABC):
    """
    Abstract base class for grouping multiple models/tools under a specific task.
    Each subclass should define a task_name and implement _create_model (or override load_model).
    """

    task_name: TaskName
    IOFormat: Tuple[str | Tuple[Any, ...], str | Tuple[Any, ...]]
    models: Dict[ModelName, Tool]

    def __init__(self):
        self.models = {}

    def get_model(self, model_name: ModelName) -> Tool:
        """
        Retrieve a loaded model by its model_name.
        If not loaded yet, calls self.load_model first.
        """
        if model_name not in self.models:
            self.load_model(model_name)
        return self.models[model_name]

    def list_models(self) -> Dict[TaskName, List]:
        """
        Return a dict showing which models are currently loaded under this task.
        """

        models_info = [model.get_model_info() for model in self.models.values()]

        return {
            self.task_name: models_info,
        }

    def _get_model_config(self, model_name: ModelName) -> ModelConfig:
        if self.task_name not in MODEL_REGISTRY:
            raise ValueError(f"Models of Task '{self.task_name}' are not implemented.")
        if model_name not in MODEL_REGISTRY[self.task_name]:
            raise ValueError(
                f"Model '{model_name}' is not implemented for Task '{self.task_name}'."
            )
        return MODEL_REGISTRY[self.task_name][model_name]

    def load_model(self, model_name: ModelName) -> None:
        """
        Load the model and options for a given model_name.
        The model process is in form of dependency injection.
        """
        if model_name in self.models:
            return

        log.opt(colors=True).info(f"Task: <blue>{self.task_name}</blue>, loading model: <green>{model_name}</green>")
        model_config = self._get_model_config(model_name)

        model, process, extra_args = self._create_model(model_name, model_config)

        self.models[model_name] = Tool(
            config=model_config, model=model, process=process, **extra_args
        )

    @abstractmethod
    def _create_model(
            self, model_name: ModelName, model_config: ModelConfig
    ) -> Tuple[Any, Callable, Dict]:
        """
        Subclasses should return a tuple of (model, process, extra_args).
        The `process` is a callable that handles how input_data is fed to the model
        and how output is returned.
        """
        ...


class SentimentAnalysisTools(GroupedTools):
    """
    Tools for sentiment analysis tasks, e.g. DistilBERT-based classification models.
    """

    def __init__(self):
        super().__init__()
        self.task_name: TaskName = "sentiment_analysis"
        self.IOFormat = ("text", "text")

    def _create_model(
            self, model_name: ModelName, model_config: ModelConfig
    ) -> Tuple[Any, Callable, Dict]:
        match model_name:
            case "distilbert-sst2" | "bert-base-multilingual" | "twitter-roberta-base":
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )

                # modify the model config to work on 2-class sentiment analysis
                model.config.id2label = {0: 'negative', 1: 'positive'}

                def process(
                        input_data: DataIncludeText, device: str
                ) -> DataIncludeText:
                    inputs = tokenizer(
                        input_data["text"], return_tensors="pt", padding=True
                    ).to(device)

                    with torch.no_grad():
                        model_output = model(**inputs)

                    # Apply softmax to get probabilities
                    probabilities = torch.softmax(model_output.logits, dim=-1)

                    prob_length = probabilities.shape[-1]
                    half = prob_length // 2
                    first_prob = probabilities.narrow(-1, 0, half).sum(dim=-1)
                    second_prob = probabilities.narrow(-1, prob_length - half, half).sum(dim=-1)
                    probabilities = torch.stack([first_prob, second_prob], dim=-1)

                    # Grab the argmax of probabilities
                    pred_ids = torch.argmax(probabilities, dim=1)
                    pred_labels = [model.config.id2label[p.item()] for p in pred_ids]
                    new_data = {"text-label": pred_labels}

                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {"tokenizer": tokenizer}

            case _:
                raise NotImplementedError(
                    f"Model '{model_name}' is not implemented for sentiment_analysis"
                )


class MachineTranslationTools(GroupedTools):
    """
    Tools for machine translation tasks, e.g. T5-based English-to-German.
    """

    def __init__(self):
        super().__init__()
        self.task_name: TaskName = "machine_translation"
        self.IOFormat = ("text", "text")

    def _create_model(
            self, model_name: ModelName, model_config: ModelConfig
    ) -> Tuple[Any, Callable, Dict]:
        match model_name:
            case "t5-base" | "t5-small" | "t5-large":
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )

                def process(
                        input_data: DataIncludeText, device: str
                ) -> DataIncludeText:
                    text_key = 'text'
                    for input_key in input_data.keys():
                        if 'text' in input_key:
                            text_key = input_key
                            break
                    text_batch = [
                        "translate English to German: " + sentence
                        for sentence in input_data[text_key]
                    ]
                    inputs = tokenizer(
                        text_batch, return_tensors="pt", padding=True, truncation=True
                    ).to(device)

                    with torch.no_grad():
                        model_output = model.generate(
                            **inputs, min_length=5, max_length=1000
                        )

                    translated_text = [
                        tokenizer.decode(output, skip_special_tokens=True)
                        for output in model_output
                    ]
                    new_data = {text_key: translated_text}

                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {"tokenizer": tokenizer}

            case _:
                raise NotImplementedError(
                    f"Model '{model_name}' is not implemented for machine_translation"
                )


class ImageClassificationTools(GroupedTools):
    """
    Tools for image classification tasks, typically using transformer-based models like ViT.
    """

    def __init__(self):
        super().__init__()
        self.task_name: TaskName = "image_classification"
        self.IOFormat = ("image", "text")

    def _create_model(
            self, model_name: ModelName, model_config: ModelConfig
    ) -> Tuple[Any, Callable, Dict]:
        match model_name:
            case "vit-base":
                processor = ViTImageProcessor.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )
                model = ViTForImageClassification.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )

                def process(
                        input_data: DataIncludeImage, device: str
                ) -> Dict[str, Any]:
                    inputs = processor(
                        images=input_data["image"], return_tensors="pt"
                    ).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    predicted_ids = outputs.logits.argmax(dim=1)
                    predicted_labels = [
                        model.config.id2label[idx.item()] for idx in predicted_ids
                    ]
                    new_data = {"text-label": predicted_labels}

                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {"processor": processor}

            case "resnet-50" | "mobilenet-v2":
                processor = AutoImageProcessor.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache, use_fast=True
                )
                model = AutoModelForImageClassification.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )

                def process(
                        input_data: DataIncludeImage, device: str
                ) -> Dict[str, Any]:
                    inputs = processor(
                        images=input_data["image"], return_tensors="pt"
                    ).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    predicted_ids = outputs.logits.argmax(dim=1)
                    predicted_labels = [
                        model.config.id2label[idx.item()] for idx in predicted_ids
                    ]
                    new_data = {"text-label": predicted_labels}

                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {"processor": processor}

            case _:
                raise NotImplementedError(
                    f"Model '{model_name}' is not implemented for image_classification"
                )


class ObjectDetectionTools(GroupedTools):
    """
    Tools for object detection tasks, e.g. DETR-based bounding box prediction.
    """

    def __init__(self):
        super().__init__()
        self.task_name: TaskName = "object_detection"
        self.IOFormat = ("image", "object_detection_information")

    def _create_model(
            self, model_name: ModelName, model_config: ModelConfig
    ) -> Tuple[Any, Callable, Dict]:
        match model_name:
            case "detr-resnet-101" | "detr-resnet-50":
                processor = AutoImageProcessor.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache, use_fast=True
                )
                model = AutoModelForObjectDetection.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache,
                )

                def process(input_data: DataIncludeImage, device: str) -> Dict[str, Any]:
                    inputs = processor(
                        images=input_data["image"], return_tensors="pt"
                    ).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)

                    threshold = 0.9
                    target_sizes = torch.tensor([
                        [inputs["pixel_values"].shape[2], inputs["pixel_values"].shape[3]]
                        for _ in range(inputs["pixel_values"].shape[0])
                    ]).to(device)
                    results = processor.post_process_object_detection(
                        outputs, target_sizes=target_sizes, threshold=threshold
                    )

                    final_outputs = []
                    predicted_results = []
                    for r in results:
                        boxes = r["boxes"].cpu().tolist()
                        scores = r["scores"].cpu().tolist()
                        labels = r["labels"].cpu().tolist()
                        label_names = [model.config.id2label[l] for l in labels]
                        final_outputs.append(
                            {"boxes": boxes, "scores": scores, "labels": label_names}
                        )
                        output = ", ".join(label_names)
                        predicted_results.append(output)

                    new_data = {"object_detection_information": final_outputs, 'text-object': predicted_results}
                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {"processor": processor}

            case "yolo-v8":
                from ultralytics import YOLO, settings

                settings.update({
                    "weights_dir": GlobalPathConfig.model_cache,
                })
                model = YOLO(os.path.join(GlobalPathConfig.model_cache, 'yolov8n.pt'))

                def process(input_data: DataIncludeImage, device: str) -> Dict[str, Any]:

                    def pad_image(images: torch.Tensor, stride=32, padding_value=114):
                        _, _, h, w = images.shape
                        total_h = (h + stride - 1) // stride * stride
                        total_w = (w + stride - 1) // stride * stride
                        pad_h = total_h - h
                        pad_w = total_w - w
                        images = F.pad(images, (0, pad_w, 0, pad_h), mode="constant", value=padding_value)
                        images = images.float() / 255.0
                        return images

                    if isinstance(input_data['image'], list):
                        input_images = torch.stack([img for img in input_data['image']]).to(device)
                    else:
                        input_images = input_data['image'].to(device)

                    inputs = pad_image(input_images).to(device)
                    results = model.predict(
                        inputs, conf=0.25, stream=False, device=device, verbose=False
                    )

                    final_outputs = []
                    predicted_results = []
                    for r in results:
                        boxes = r.boxes.data.cpu().tolist()
                        # boxes is 6-dim tensor: [x1, y1, x2, y2, conf, cls], remove the last two dims
                        boxes = [box[:4] for box in boxes]
                        scores = r.boxes.conf.cpu().tolist()
                        labels = r.boxes.cls.cpu().tolist()
                        label_names = [r.names[int(l)] for l in labels]

                        final_outputs.append(
                            {"boxes": boxes, "scores": scores, "labels": label_names}
                        )
                        output = ", ".join(label_names)
                        predicted_results.append(output)

                    new_data = {"object_detection_information": results, 'text-object': predicted_results}
                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {}

            case _:
                raise NotImplementedError(
                    f"Model '{model_name}' is not implemented for object_detection"
                )


class ImageSuperResolutionTools(GroupedTools):
    """
    Tools for image super-resolution tasks, e.g. Swin2SR.
    """

    def __init__(self):
        super().__init__()
        self.task_name: TaskName = "image_super_resolution"
        self.IOFormat = ("image", "image")

    def _create_model(
            self, model_name: ModelName, model_config: ModelConfig
    ) -> Tuple[Any, Callable, Dict]:
        match model_name:
            case "swin2sr":
                processor = AutoImageProcessor.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache, use_fast=False
                )
                model = Swin2SRForImageSuperResolution.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )

                def process(
                        input_data: DataIncludeImage, device: str
                ) -> DataIncludeImage:
                    inputs = processor(
                        images=input_data["image"], return_tensors="pt"
                    ).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)

                    # outputs.reconstruction is shape: (B, C, H, W)
                    sr_images = []
                    reconstructions = outputs.reconstruction.clamp_(0, 1)
                    for i in range(reconstructions.shape[0]):
                        out_img = reconstructions[i]  # .permute(1, 2, 0)
                        out_img = (out_img.cpu().numpy() * 255.0).round().astype(np.uint8)
                        sr_images.append(torch.from_numpy(out_img))

                    new_data = {"image": sr_images}
                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {"processor": processor}

            case _:
                raise NotImplementedError(
                    f"Model '{model_name}' is not implemented for image_super_resolution"
                )


class ImageColorizationTools(GroupedTools):
    """
    Tools for image colorization tasks, e.g. Siggraph17-based colorization.
    """

    def __init__(self):
        super().__init__()
        self.task_name: TaskName = "image_colorization"
        self.IOFormat = ("image", "image")

    def _create_model(
            self, model_name: ModelName, model_config: ModelConfig
    ) -> Tuple[Any, Callable, Dict]:
        match model_name:
            case "siggraph17" | "eccv16":
                from .github_models.colorization import colorizers
                import cv2

                preprocess_img = colorizers.preprocess_img
                postprocess_tens = colorizers.postprocess_tens

                if model_name == "siggraph17":
                    model = colorizers.siggraph17()
                else:
                    model = colorizers.eccv16()

                def process(
                        input_data: DataIncludeImage, device: str
                ) -> DataIncludeImage:
                    colorized_images = []
                    for img in input_data["image"]:
                        # Assume each img is shape: (C,H,W) with C channel
                        img_np = img.permute(1, 2, 0).cpu().numpy()
                        tens_l_orig, tens_l_rs = preprocess_img(img_np, HW=(256, 256))
                        tens_l_rs = tens_l_rs.to(device)

                        with torch.no_grad():
                            out_ab = model(tens_l_rs).cpu()

                        out_img = postprocess_tens(tens_l_orig, out_ab)
                        out_img = cv2.normalize(
                            out_img,
                            None,
                            alpha=0,
                            beta=255,
                            norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_32F,
                        ).astype(np.uint8)

                        # Convert back to torch.Tensor with shape (C,H,W)
                        color_tensor = torch.from_numpy(out_img).permute(2, 0, 1)
                        colorized_images.append(color_tensor)

                    new_data = {"image": colorized_images}
                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {}

            case "icolorit":
                from einops import rearrange
                from timm.models import create_model  # timm>=0.6
                from .github_models.iColoriT import modeling
                from .github_models.iColoriT.utils import lab2rgb, rgb2lab

                model_name = "icolorit_base_4ch_patch16_224"
                model = create_model(
                    model_name,
                    pretrained=False,
                    drop_path_rate=0.0,
                    use_rpb=True,
                    avg_hint=True,
                    head_mode="cnn",
                    mask_cent=False,
                )

                icolorit_root = f"{os.path.dirname(os.path.abspath(__file__))}/github_models/iColoriT"
                ckpt_path = f"{icolorit_root}/pretrained_models/{model_name}.pth"
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict(state["model"] if "model" in state else state, strict=True)

                patch_size = model.patch_embed.patch_size  # (16,16)
                input_size = 224

                def process(input_data: DataIncludeImage, device: str) -> DataIncludeImage:
                    coloured = []

                    for g in input_data["image"]:
                        if g.shape[0] == 1:
                            g_rgb = g.repeat(3, 1, 1)
                        else:
                            g_rgb = g[:3]
                        H0, W0 = g_rgb.shape[1:]
                        img = g_rgb.float().div(255.)
                        img = F.interpolate(img.unsqueeze(0), size=input_size,
                                            mode="bilinear", align_corners=False)

                        img_lab = rgb2lab(img.clone().to(device))
                        h = w = input_size // patch_size[0]
                        bool_hint = torch.zeros(1, h * w, dtype=torch.bool, device=device)

                        with torch.inference_mode(), torch.amp.autocast(device_type='cuda'):
                            out = model(img_lab, bool_hint)

                        p1, p2 = patch_size
                        out = rearrange(out, 'b n (p1 p2 c) -> b n (p1 p2) c',
                                        p1=p1, p2=p2, c=2)
                        lab_patches = rearrange(img_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c',
                                                p1=p1, p2=p2)
                        pred_lab = torch.cat([lab_patches[..., :1], out], dim=3)
                        pred_lab = rearrange(pred_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                                             h=h, w=w, p1=p1, p2=p2)  # (1,3,224,224)

                        pred_rgb = lab2rgb(pred_lab).clamp(0, 1)  # float[0,1]
                        if (H0, W0) != (input_size, input_size):
                            pred_rgb = F.interpolate(pred_rgb, size=(H0, W0),
                                                     mode="bilinear", align_corners=False)

                        pred_uint8 = img_as_ubyte(pred_rgb[0].permute(1, 2, 0).detach().cpu().numpy())
                        coloured.append(torch.from_numpy(pred_uint8).permute(2, 0, 1))

                    new_data = input_data.copy()
                    new_data["image"] = coloured
                    return new_data

                return model, process, {}

            case _:
                raise NotImplementedError(
                    f"Model '{model_name}' is not implemented for image_colorization"
                )


class ImageDenoisingTools(GroupedTools):
    """
    Tools for image denoising, e.g. Restormer or other custom denoise models.
    """

    def __init__(self):
        super().__init__()
        self.task_name: TaskName = "image_denoising"
        self.IOFormat = ("image", "image")

    def _create_model(
            self, model_name: ModelName, model_config: ModelConfig
    ) -> Tuple[Any, Callable, Dict]:
        match model_name:
            case "restormer-denoise":
                current_dir = os.path.dirname(os.path.abspath(__file__))
                restormer_arch_path = f"{current_dir}/github_models/Restormer/basicsr/models/archs/restormer_arch.py"
                arch_dict = run_path(restormer_arch_path)
                Restormer = arch_dict["Restormer"]

                # initialize model parameters
                parameters = {
                    "inp_channels": 3,
                    "out_channels": 3,
                    "dim": 48,
                    "num_blocks": [4, 6, 6, 8],
                    "num_refinement_blocks": 4,
                    "heads": [1, 2, 4, 8],
                    "ffn_expansion_factor": 2.66,
                    "bias": False,
                    "LayerNorm_type": "BiasFree",  # note: denoising often uses 'BiasFree'
                    "dual_pixel_task": False,
                }
                model = Restormer(**parameters)
                ckpt_path = f"{current_dir}/github_models/Restormer/Denoising/pretrained_models/real_denoising.pth"
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint["params"])

                def process(
                        input_data: DataIncludeImage, device: str
                ) -> DataIncludeImage:
                    """
                    Assume input_data["image"] is List[torch.Tensor], each tensor shape (C,H,W), range 0~255.
                    Output: same structure, List[torch.Tensor], shape (C,H,W), range 0~255.
                    """
                    restored_images: List[torch.Tensor] = []
                    img_multiple_of = 8

                    with torch.no_grad():
                        for img_t in input_data["image"]:
                            # 1) convert to (1, C, H, W), normalize to [0,1]
                            c, h, w = img_t.shape
                            inp = img_t.float().div(255.0).unsqueeze(0).to(device)

                            # 2) padding
                            Hpad = (h + img_multiple_of) // img_multiple_of * img_multiple_of
                            Wpad = (w + img_multiple_of) // img_multiple_of * img_multiple_of
                            padh = Hpad - h
                            padw = Wpad - w
                            inp = F.pad(inp, (0, padw, 0, padh), mode="reflect")

                            # 3) forward
                            denoised = model(inp)
                            denoised = torch.clamp(denoised, 0, 1)

                            # 4) remove pad
                            denoised = denoised[:, :, :h, :w]

                            # 5) convert to [0,255] integer tensor
                            denoised_np = denoised[0].permute(1, 2, 0).cpu().numpy()
                            denoised_np = img_as_ubyte(denoised_np)  # (H,W,C) uint8
                            # 6) convert back to (C,H,W)
                            restored_t = torch.from_numpy(denoised_np).permute(2, 0, 1)
                            restored_images.append(restored_t)

                    new_data = {"image": restored_images}
                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {}

            case "MPRNet":
                from .github_models.MPRNet.demo import load_model
                model = load_model("Denoising")
                img_multiple_of = 8

                def process(
                        input_data: DataIncludeImage, device: str
                ) -> DataIncludeImage:
                    restored_images: List[torch.Tensor] = []

                    with torch.no_grad():
                        for img_t in input_data["image"]:
                            c, h, w = img_t.shape
                            inp = img_t.float().div(255.0).unsqueeze(0).to(device)

                            Hpad = (h + img_multiple_of) // img_multiple_of * img_multiple_of
                            Wpad = (w + img_multiple_of) // img_multiple_of * img_multiple_of
                            padh, padw = Hpad - h, Wpad - w
                            inp = F.pad(inp, (0, padw, 0, padh), mode="reflect")

                            denoised = model(inp)  # shape: (1,3,HpWp)

                            if isinstance(denoised, (list, tuple)):
                                denoised = denoised[-1]
                            denoised = torch.clamp(denoised, 0, 1)

                            denoised = denoised[:, :, :h, :w]

                            # (1,3,H,W) -> (H,W,C)
                            den_np = denoised[0].permute(1, 2, 0).cpu().numpy()
                            den_np = img_as_ubyte(den_np)  # [0,255] uint8

                            # 回到 (C,H,W) torch.uint8
                            restored_t = torch.from_numpy(den_np).permute(2, 0, 1)
                            restored_images.append(restored_t)

                    new_data = {"image": restored_images}
                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {}

            case "swin-ir":
                import requests
                from .github_models.SwinIR.models.network_swinir import SwinIR

                model = SwinIR(
                    upscale=1,
                    in_chans=3,
                    img_size=128,
                    window_size=8,
                    img_range=1.,
                    depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2,
                    upsampler='',
                    resi_connection='1conv',
                )

                swinir_root = f"{os.path.dirname(os.path.abspath(__file__))}/github_models/SwinIR"
                ckpt_name = "005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth"
                ckpt_path = f"{swinir_root}/pretrained_models/{ckpt_name}"

                if not os.path.exists(ckpt_path):
                    url = (
                        "https://github.com/JingyunLiang/SwinIR/releases/"
                        f"download/v0.0/{ckpt_name}"
                    )
                    log.info(f"[SwinIR] downloading weights from {url}")
                    r = requests.get(url, allow_redirects=True, timeout=60)
                    open(ckpt_path, "wb").write(r.content)

                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt["params"] if "params" in ckpt else ckpt, strict=True)

                window_size = 8

                def process(input_data: DataIncludeImage, device: str) -> DataIncludeImage:
                    restored: List[torch.Tensor] = []

                    with torch.no_grad():
                        for img_t in input_data["image"]:
                            if img_t.shape[0] == 1:
                                img_t = img_t.repeat(3, 1, 1)

                            c, h, w = img_t.shape
                            inp = img_t.float().div(255.0).unsqueeze(0).to(device)

                            h_pad = (h // window_size + 1) * window_size - h
                            w_pad = (w // window_size + 1) * window_size - w
                            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h + h_pad, :]
                            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w + w_pad]

                            out = model(inp)
                            out = out[..., :h, :w].clamp_(0, 1)

                            out_np = out[0].permute(1, 2, 0).cpu().numpy()  # (H,W,C), float
                            out_np = img_as_ubyte(out_np)  # uint8
                            restored.append(torch.from_numpy(out_np).permute(2, 0, 1))

                    new_data = {"image": restored}
                    updated = input_data.copy()
                    updated.update(new_data)
                    return updated

                return model, process, {}

            case _:
                raise NotImplementedError(
                    f"Model '{model_name}' is not implemented for image_denoising"
                )


class ImageDeblurringTools(GroupedTools):
    """
    Tools for image deblurring, e.g. Restormer-based defocus deblurring.
    """

    def __init__(self):
        super().__init__()
        self.task_name: TaskName = "image_deblurring"
        self.IOFormat = ("image", "image")

    def _create_model(
            self, model_name: ModelName, model_config: ModelConfig
    ) -> Tuple[Any, Callable, Dict]:
        match model_name:
            case "restormer-deblur":

                # 1) load Restormer architecture
                current_dir = os.path.dirname(os.path.abspath(__file__))
                restormer_arch_path = f"{current_dir}/github_models/Restormer/basicsr/models/archs/restormer_arch.py"
                arch_dict = run_path(restormer_arch_path)
                Restormer = arch_dict["Restormer"]

                # 2) initialize model parameters
                parameters = {
                    "inp_channels": 3,
                    "out_channels": 3,
                    "dim": 48,
                    "num_blocks": [4, 6, 6, 8],
                    "num_refinement_blocks": 4,
                    "heads": [1, 2, 4, 8],
                    "ffn_expansion_factor": 2.66,
                    "bias": False,
                    "LayerNorm_type": "WithBias",  # defocus_deblurring may use 'WithBias'
                    "dual_pixel_task": False,
                }
                model = Restormer(**parameters)

                # 3) load pretrained weights
                ckpt_path = f"{current_dir}/github_models/Restormer/Defocus_Deblurring/pretrained_models/single_image_defocus_deblurring.pth"
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint["params"])

                def process(
                        input_data: DataIncludeImage, device: str
                ) -> DataIncludeImage:
                    restored_images: List[torch.Tensor] = []
                    img_multiple_of = 8

                    with torch.no_grad():
                        for img_t in input_data["image"]:
                            # same procedure as denoise
                            c, h, w = img_t.shape
                            inp = img_t.float().div(255.0).unsqueeze(0).to(device)

                            # pad
                            Hpad = (h + img_multiple_of) // img_multiple_of * img_multiple_of
                            Wpad = (w + img_multiple_of) // img_multiple_of * img_multiple_of
                            padh = Hpad - h
                            padw = Wpad - w
                            inp = F.pad(inp, (0, padw, 0, padh), mode="reflect")

                            # forward
                            deblurred = model(inp)
                            deblurred = torch.clamp(deblurred, 0, 1)

                            # remove pad
                            deblurred = deblurred[:, :, :h, :w]

                            # convert back to (C,H,W) in 0-255
                            deblurred_np = deblurred[0].permute(1, 2, 0).cpu().numpy()
                            deblurred_np = img_as_ubyte(deblurred_np)
                            restored_t = torch.from_numpy(deblurred_np).permute(2, 0, 1)
                            restored_images.append(restored_t)

                    new_data = {"image": restored_images}
                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {}

            case "MPRNet":
                from .github_models.MPRNet.demo import load_model
                model = load_model("Deblurring")
                img_multiple_of = 8

                def process(
                        input_data: DataIncludeImage, device: str
                ) -> DataIncludeImage:
                    restored_images: List[torch.Tensor] = []

                    with torch.no_grad():
                        for img_t in input_data["image"]:
                            c, h, w = img_t.shape
                            inp = img_t.float().div(255.0).unsqueeze(0).to(device)

                            Hpad = (h + img_multiple_of) // img_multiple_of * img_multiple_of
                            Wpad = (w + img_multiple_of) // img_multiple_of * img_multiple_of
                            padh, padw = Hpad - h, Wpad - w
                            inp = F.pad(inp, (0, padw, 0, padh), mode="reflect")

                            denoised = model(inp)  # shape: (1,3,HpWp)
                            if isinstance(denoised, (list, tuple)):
                                denoised = denoised[-1]

                            denoised = torch.clamp(denoised, 0, 1)

                            denoised = denoised[:, :, :h, :w]

                            # (1,3,H,W) -> (H,W,C)
                            den_np = denoised[0].permute(1, 2, 0).cpu().numpy()
                            den_np = img_as_ubyte(den_np)  # [0,255] uint8

                            # 回到 (C,H,W) torch.uint8
                            restored_t = torch.from_numpy(den_np).permute(2, 0, 1)
                            restored_images.append(restored_t)

                    new_data = {"image": restored_images}
                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {}

            case "deblurgan-v2":
                from .github_models.DeblurGANv2.predict import Predictor

                predictor = Predictor(cache_dir=GlobalPathConfig.model_cache)
                model = predictor.model

                def process(
                        input_data: DataIncludeImage, device: str
                ) -> DataIncludeImage:
                    # GAN inference should be in train mode to use actual stats in norm layers,
                    # it's not a bug
                    model.train(True)
                    restored_images: List[torch.Tensor] = []
                    for img_tensor in input_data["image"]:
                        img_np = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        pred_np = predictor(img_np, mask=None, device=device)

                        restored_tensor = torch.from_numpy(pred_np).permute(2, 0, 1)
                        restored_images.append(restored_tensor)
                    new_data = {"image": restored_images}
                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {}

            case _:
                raise NotImplementedError(
                    f"Model '{model_name}' is not implemented for image_deblurring"
                )


class ImageCaptioningTools(GroupedTools):
    """
    Tools for image captioning, e.g. using a VisionEncoderDecoderModel.
    """

    def __init__(self):
        super().__init__()
        self.task_name: TaskName = "image_captioning"
        self.IOFormat = ("image", "text")

    def _create_model(
            self, model_name: ModelName, model_config: ModelConfig
    ) -> Tuple[Any, Callable, Dict]:
        match model_name:
            case "vit-gpt2":
                processor = ViTImageProcessor.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )
                model = VisionEncoderDecoderModel.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )

                def process(input_data: DataIncludeImage, device: str) -> DataIncludeText:
                    pixel_values = processor(
                        images=input_data["image"], return_tensors="pt"
                    ).pixel_values.to(device)
                    with torch.no_grad():
                        output_ids = model.generate(
                            pixel_values=pixel_values, max_length=40, num_beams=4
                        )
                    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    captions = [p.strip() for p in preds]
                    new_data = {"text-caption": captions}
                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {"processor": processor, "tokenizer": tokenizer}

            case "blip-large":
                # Use BLIP-specific processor and model
                from transformers import BlipProcessor, BlipForConditionalGeneration
                processor = BlipProcessor.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )
                model = BlipForConditionalGeneration.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )
                # BLIP processor includes a tokenizer internally
                tokenizer = processor.tokenizer

                def process(input_data: DataIncludeImage, device: str) -> DataIncludeText:
                    # Convert input image tensor to pixel values
                    pixel_values = processor(
                        images=input_data["image"], return_tensors="pt"
                    ).pixel_values.to(device)
                    with torch.no_grad():
                        output_ids = model.generate(
                            pixel_values=pixel_values, max_length=40, num_beams=4
                        )
                    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    captions = [p.strip() for p in preds]
                    new_data = {"text-caption": captions}
                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {"processor": processor, "tokenizer": tokenizer}

            case "git-base-coco":
                # Use GIT processor and model (processor wraps CLIP image processor + BERT tokenizer)
                processor = AutoProcessor.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )
                # Load GIT as a causal LM model
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )
                tokenizer = processor.tokenizer  # GitProcessor's tokenizer (BERT-based)

                def process(input_data: DataIncludeImage, device: str) -> DataIncludeText:
                    pixel_values = processor(
                        images=input_data["image"], return_tensors="pt"
                    ).pixel_values.to(device)
                    with torch.no_grad():
                        output_ids = model.generate(
                            pixel_values=pixel_values, max_length=40, num_beams=4
                        )
                    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    captions = [p.strip() for p in preds]
                    new_data = {"text-caption": captions}
                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {"processor": processor, "tokenizer": tokenizer}

            case _:
                raise NotImplementedError(
                    f"Model '{model_name}' is not implemented for image_captioning"
                )


class TextToImageTools(GroupedTools):
    """
    Tools for text-to-image generation, e.g. Stable Diffusion or other diffusion-based models.
    """

    def __init__(self):
        super().__init__()
        self.task_name: TaskName = "text_to_image"
        self.IOFormat = ("text", "image")

    def _create_model(
            self, model_name: ModelName, model_config: ModelConfig
    ) -> Tuple[Any, Callable, Dict]:
        # todo: implement text-to-image generation
        # from diffusers import StableDiffusionPipeline

        # # need a dummy function to avoid SafetyChecker error (similar to old code)
        # def dummy_safety_checker(images, clip_input):
        #     return images, False

        match model_name:
            #     case "stable-diffusion-v1-4":
            #         pipe = StableDiffusionPipeline.from_pretrained(
            #             model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
            #         )
            #         # remove safety checker
            #         pipe.safety_checker = dummy_safety_checker
            #         pipe.enable_attention_slicing()

            #         def process(
            #             input_data: DataIncludeText, device: str
            #         ) -> DataIncludeImage:
            #             """
            #             input_data: {"text": List[str]}  # multiple prompts
            #             returns: {"image": List[torch.Tensor]} or PIL
            #             """
            #             pipe.to(device)
            #             images = []
            #             for prompt in input_data["text"]:
            #                 with torch.no_grad():
            #                     out = pipe(prompt).images[0]  # generate one image
            #                 # out is PIL Image; if you want tensor, you can convert it yourself
            #                 # here directly convert to tensor (C,H,W) 0~255
            #                 img_t = transforms.ToTensor()(out) * 255.0
            #                 img_t = img_t.byte()
            #                 images.append(img_t)

            #             return {"image": images}

            #         return pipe, process, {}

            case _:
                raise NotImplementedError(
                    f"Model '{model_name}' is not implemented for text_to_image"
                )


class QuestionAnsweringTools(GroupedTools):
    """
    Tools for text-based question answering, e.g. DistilBERT trained on SQuAD.
    IOFormat is recommended to be (("text","text"), "text") to represent (contexts, questions) -> answers
    """

    def __init__(self):
        super().__init__()
        self.task_name: TaskName = "question_answering"
        self.IOFormat = (("text", "text"), "text")

    def _create_model(
            self, model_name: ModelName, model_config: ModelConfig
    ) -> Tuple[Any, Callable, Dict]:
        match model_name:
            # case "distilbert-squad":
            #     tokenizer = AutoTokenizer.from_pretrained(
            #         model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
            #     )
            #     model = AutoModelForQuestionAnswering.from_pretrained(
            #         model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
            #     )

            #     def process(
            #         input_data: Dict[str, List[str]], device: str
            #     ) -> Dict[str, List[str]]:
            #         """
            #         expect input_data = {"context": [...], "question": [...]}
            #         or you can follow the old code, put context/question in text[0], text[1]?
            #         here for readability, assume:
            #             input_data["context"]  = List[str]
            #             input_data["question"] = List[str]
            #         """
            #         contexts = input_data["context"]
            #         questions = input_data["question"]

            #         # if the number of contexts and questions do not match, you need to handle it yourself or raise an error
            #         if len(contexts) != len(questions):
            #             raise ValueError("Number of contexts and questions must match.")

            #         inputs = tokenizer(
            #             questions,
            #             contexts,
            #             return_tensors="pt",
            #             padding=True,
            #             truncation=True,
            #         ).to(device)

            #         with torch.no_grad():
            #             outputs = model(**inputs)

            #         start_logits = outputs.start_logits
            #         end_logits = outputs.end_logits

            #         answers = []
            #         for i in range(len(questions)):
            #             start_idx = torch.argmax(start_logits[i]).item()
            #             end_idx = torch.argmax(end_logits[i]).item()
            #             # decode the token from start:end+1
            #             token_ids = inputs["input_ids"][i][start_idx : end_idx + 1]
            #             ans_str = tokenizer.decode(token_ids, skip_special_tokens=True)
            #             answers.append(ans_str)
            #
            #         new_data = {"text": answers}
            #         updated_data = input_data.copy()
            #         updated_data.update(new_data)
            #         return updated_data
            #
            #     return model, process, {"tokenizer": tokenizer}

            case _:
                raise NotImplementedError(
                    f"Model '{model_name}' is not implemented for question_answering"
                )


class VisualQuestionAnsweringTools(GroupedTools):
    """
    Tools for visual question answering (VQA), e.g. using ViLT.
    """

    def __init__(self):
        super().__init__()
        self.task_name: TaskName = "visual_question_answering"

    def _create_model(
            self, model_name: ModelName, model_config: ModelConfig
    ) -> Tuple[Any, Callable, Dict]:
        match model_name:
            case "vilt-vqa":
                processor = ViltProcessor.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )
                model = ViltForQuestionAnswering.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )

                def process(input_data: DataIncludeImageAndText, device: str) -> DataIncludeText:
                    """
                    input_data:
                      {
                        'images': List[torch.Tensor],
                        'text': List[str] # questions
                      }
                      They must have the same length or be broadcastable.
                    returns: List[str] predicted answers.
                    """
                    images = input_data["image"]
                    questions = input_data["text"]
                    enc = processor(
                        images, questions, return_tensors="pt", padding=True
                    ).to(device)
                    with torch.no_grad():
                        outputs = model(**enc)

                    idxs = torch.argmax(outputs.logits, dim=1)
                    answers = [model.config.id2label[idx.item()] for idx in idxs]

                    new_data = {"text": answers}
                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {"processor": processor}

            case _:
                raise NotImplementedError(
                    f"Model '{model_name}' is not implemented for visual_question_answering"
                )


class TextSummarizationTools(GroupedTools):
    """
    Tools for text summarization tasks, e.g. BART on CNN/DailyMail dataset.
    """

    def __init__(self):
        super().__init__()
        self.task_name: TaskName = "text_summarization"

    def _create_model(
            self, model_name: ModelName, model_config: ModelConfig
    ) -> Tuple[Any, Callable, Dict]:
        match model_name:
            case "bart-cnn":
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )

                def process(input_data: DataIncludeText, device: str) -> DataIncludeText:
                    """
                    input_data: {'text': List[str]}, each item is a passage to summarize.
                    """
                    inputs = tokenizer(
                        input_data["text"],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(device)

                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_length=142, min_length=56)

                    summaries = [
                        tokenizer.decode(g, skip_special_tokens=True) for g in outputs
                    ]
                    new_data = {"text": summaries}

                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {"tokenizer": tokenizer}

            case _:
                raise NotImplementedError(
                    f"Model '{model_name}' is not implemented for text_summarization"
                )


class TextGenerationTools(GroupedTools):
    """
    Tools for text generation tasks, e.g. GPT-2-based generative models.
    """

    def __init__(self):
        super().__init__()
        self.task_name: TaskName = "text_generation"

    def _create_model(
            self, model_name: ModelName, model_config: ModelConfig
    ) -> Tuple[Any, Callable, Dict]:
        match model_name:
            case "gpt2-base":
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )
                # Add a pad token to avoid warnings during generate
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

                model = AutoModelForCausalLM.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )
                model.resize_token_embeddings(len(tokenizer))

                def process(input_data: DataIncludeText, device: str) -> DataIncludeText:
                    """
                    input_data: {'text': List[str]}
                    returns: {'text': List[str]}
                    """
                    results = []
                    for text_item in input_data["text"]:
                        inputs = tokenizer(text_item, return_tensors="pt").to(device)

                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                min_length=5,
                                max_length=30,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        results.append(gen_text)

                    new_data = {"text": results}
                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {"tokenizer": tokenizer}

            case _:
                raise NotImplementedError(
                    f"Model '{model_name}' is not implemented for text_generation"
                )


class MaskFillingTools(GroupedTools):
    """
    Tools for masked language modeling (fill in the blank), e.g. DistilBERT MLM.
    """

    def __init__(self):
        super().__init__()
        self.task_name: TaskName = "mask_filling"

    def _create_model(
            self, model_name: ModelName, model_config: ModelConfig
    ) -> Tuple[Any, Callable, Dict]:
        match model_name:
            case "distilbert-mlm":
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )
                model = AutoModelForMaskedLM.from_pretrained(
                    model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache
                )

                def process(input_data: DataIncludeText, device: str) -> DataIncludeText:
                    """
                    input_data: {'text': List[str]} containing [MASK] tokens.
                    returns: {'text': List[str]} with [MASK] replaced by the top-1 predicted token.
                    """
                    inputs = tokenizer(
                        input_data["text"], return_tensors="pt", padding=True
                    ).to(device)

                    with torch.no_grad():
                        outputs = model(**inputs)

                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    results = []
                    mask_token_id = tokenizer.mask_token_id

                    for i, original_text in enumerate(input_data["text"]):
                        mask_positions = (inputs.input_ids[i] == mask_token_id).nonzero()
                        if len(mask_positions) == 0:
                            # If no [MASK], just append original text
                            results.append(original_text)
                            continue

                        # For simplicity, assume only one [MASK] per sequence
                        masked_index = mask_positions[0].item()
                        token_probs = probabilities[i, masked_index]
                        top_token_id = torch.topk(token_probs, k=1).indices[0].item()
                        predicted_token = tokenizer.convert_ids_to_tokens([top_token_id])[0]

                        # Replace the first occurrence of [MASK] in the text
                        filled_text = original_text.replace(
                            tokenizer.mask_token, predicted_token, 1
                        )
                        results.append(filled_text)

                    new_data = {"text": results}
                    updated_data = input_data.copy()
                    updated_data.update(new_data)
                    return updated_data

                return model, process, {"tokenizer": tokenizer}

            case _:
                raise NotImplementedError(
                    f"Model '{model_name}' is not implemented for mask_filling"
                )
