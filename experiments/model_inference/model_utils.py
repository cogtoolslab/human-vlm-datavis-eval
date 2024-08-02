from .model_helpers.deplot import llm_prompts
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoProcessor
from transformers import LlavaForConditionalGeneration
from transformers import Pix2StructForConditionalGeneration

from transformers import BitsAndBytesConfig
from transformers import GenerationConfig
import subprocess

import torch

# device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
# device = get_device_with_most_free_memory()
device = "cuda:2"
torch.cuda.set_device(device)
print("USING DEVICE: ", device)

# from openai import OpenAI
from .data_utils import DataUtils
from .api.azure_gpt4v import gpt4v

def llava_cli(agent, image_url, prompt, temperature, top_p):
    program = "/home/arnavv/miniconda3/envs/llava/bin/python"
    # args = f"-m llava.eval.run_llava --model-path '{agent}' --image-file '{image_url}' --query '{prompt}' --temperature '{temperature}' --top_p '{top_p}'"
    args = [
        "-m", "llava.eval.run_llava",
        "--model-path", agent,
        "--image-file", image_url,
        "--query", prompt,
        "--temperature", str(temperature),
        "--top_p", str(top_p),
        "--device", device
    ]
    result = subprocess.run([program] + args, capture_output=True, text=True)
    # print(result.stderr)
    # result = subprocess.run([program, args], capture_output=True, text=True)
    return result.stdout

class ModelUtils:
    def __init__(self, agentType, temperature, topP):
        self.agentType = agentType
        self.temperature = temperature
        self.topP = topP

        if ((self.agentType == "Salesforce/blip2-opt-2.7b") or 
            (self.agentType == "Salesforce/blip2-opt-6.7b") or 
            (self.agentType == "Salesforce/blip2-flan-t5-xxl") or 
            (self.agentType == "Salesforce/blip2-flan-t5-xl")):
            self.model, self.processor = self.load_blip2()

        elif (self.agentType == "llava-hf/llava-1.5-7b-hf" or self.agentType == "llava-hf/llava-1.5-13b-hf"):
            self.model, self.processor = self.load_llava()

        elif ((self.agentType == "google/pix2struct-chartqa-base") or 
              self.agentType == "google/matcha-chartqa" or 
              self.agentType == "google/deplot"):

            self.model, self.processor = self.load_pix2struct()

    def process_image(self, image_url, prompt, max_length=270, incontext_examples=None):

        if self.agentType == "GPT-4V":
            return gpt4v(image_url, prompt, self.temperature, self.topP, incontext_examples)

        elif self.agentType.find("liuhaotian") != -1:
            return llava_cli(self.agentType, image_url, prompt, self.temperature, self.topP)

        elif self.agentType == "google/deplot+gpt4":
            image = DataUtils.get_image(image_url)
            return deplot(image, image_url)

        else: # default to hugging face 
            image = DataUtils.get_image(image_url)
            if (self.agentType == "llava-hf/llava-1.5-7b-hf" or self.agentType == "llava-hf/llava-1.5-13b-hf"):
                inputs = self.processor(text=prompt, 
                                        images=image, 
                                        return_tensors="pt"
                                        ).to(device)
            else:
                inputs = self.processor(image, 
                                        text=prompt, 
                                        return_tensors="pt",
                                        ).to(device, torch.float16)

            generation_config = GenerationConfig(
                do_sample=True,
                temperature=self.temperature,
                top_p=self.topP,
                max_new_tokens=max_length,   #max_length=max_length,
            )
            generated_ids = self.model.generate(**inputs,  do_sample=True,
                temperature=self.temperature,
                top_p=self.topP,
                max_new_tokens=max_length)#generation_config=generation_config)

            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0].strip()
                

            return generated_text


    def load_blip2(self):
        quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
        )
        blip_model = Blip2ForConditionalGeneration.from_pretrained(
            self.agentType, 
            #device_map="auto",
            quantization_config=quantization_config
        )
        
        blip_processor = Blip2Processor.from_pretrained(self.agentType)

        return blip_model, blip_processor
    
    def load_llava(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        processor = AutoProcessor.from_pretrained(self.agentType)
        model = LlavaForConditionalGeneration.from_pretrained(
            self.agentType,
            quantization_config=quantization_config,
            # device_map="auto"
            )

        return model, processor

    def load_pix2struct(self):
        quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
        )
        processor = AutoProcessor.from_pretrained(self.agentType)
        model = Pix2StructForConditionalGeneration.from_pretrained(self.agentType,
                                                                   quantization_config=quantization_config)

        return model, processor


    def process_blip2(self, image, prompt, model, processor):
        inputs = processor(image, text=prompt, return_tensors="pt").to(self.config.device, torch.float16)
        generated_ids = model.generate(**inputs, )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text
