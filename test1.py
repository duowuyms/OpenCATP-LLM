from src.tools.tool_manager import tool_manager
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7895'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7895'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

tool_manager.load_models('image_captioning')
