# from src.tools import tool_manager
from src.config import GlobalPathConfig as pCfg
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoProcessor,
    AutoModelForImageTextToText,
    logging, AutoModelForCausalLM,
)
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
from src.tools.tool_manager import tool_manager
from src.utils import get_model_parameters_size
import os
from ultralytics import YOLO

# from ultralytics.data.loaders import LoadTensor

# # logging.set_verbosity_debug()
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7895'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7895'
# os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

nowID = '1-3'
device = 'cuda:0'
data_path = './dataset/111/inputs/text.txt'
label_path = './dataset/111/outputs/labels.txt'

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.splitlines()

with open(label_path, 'r', encoding='utf-8') as f:
    truth = f.readlines()
    truth = [i.strip() for i in truth]

image_path = './dataset/201/inputs/images/0.jpg'
image_preprocessor = transforms.Compose([
    transforms.PILToTensor(),
])
with Image.open(image_path) as img:
    img = image_preprocessor(img)
    img = img.unsqueeze(0)

match nowID:
    case '1-1':
        tool = tool_manager.get_model('sentiment_analysis', 'twitter-roberta-base')
        tool.to(device)
        result = tool.execute({
            "text": text,
        })
        print('model-size', get_model_parameters_size(tool.model))
        print(result['text-label'])
        print(truth)

    case '1-2':

        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest",
                                                  cache_dir=pCfg.hf_cache)
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest",
                                                                   cache_dir=pCfg.hf_cache)
        model = model.to(device)
        model.config.id2label = {0: 'negative', 1: 'positive'}

        inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
        result = model(**inputs).logits

        half = result.shape[1] // 2
        first_prob = result.narrow(1, 0, half).sum(dim=1)
        second_prob = result.narrow(1, half, half).sum(dim=1)

        result = torch.stack([first_prob, second_prob], dim=1)

        label_id = torch.argmax(result, dim=1)
        labels = [model.config.id2label[i.item()] for i in label_id]
        print(result)
        print(model.config.id2label)

    case '1-3':
        model = YOLO("yolov8n.pt")
        model.to(device)
        model.eval()


        def pad_img(img, stride=32, padding_value=114):
            _, _, h, w = img.shape
            pad_h = (h + stride - 1) // stride * stride
            pad_w = (w + stride - 1) // stride * stride
            pad_h = pad_h - h
            pad_w = pad_w - w
            pad_h = pad_h // 2
            pad_w = pad_w // 2
            img = F.pad(img, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=padding_value)
            return img


        img = img.float()
        result = model.predict(
            source=pad_img(img),
            conf=0.5,
            stream=False
        )
        result[0].show()
        print(result)
