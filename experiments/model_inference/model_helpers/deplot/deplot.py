from .model_helpers.deplot.deplot import llm_prompts, TemplateKey
from transformers import Pix2StructForConditionalGeneration
from .api.azure_gpt4 import gpt4
import requests
import torch

# device = "cuda:4"
# torch.cuda.set_device(device)
def deplot(image, prompt, temperature, top_p):
    quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
        )
    deplot_processor = AutoProcessor.from_pretrained(agentType)
    deplot_model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot",
                                                                quantization_config=quantization_config)
    
    resoponse_table = deplot_processor(
        image, 
        text="Generate underlying data table of the figure below:", 
        return_tensors="pt",
    ).to(device, torch.float16)

    generation_config = GenerationConfig(
                max_new_tokens=1024
            )

    generated_ids = deplot_model.generate(**inputs, generation_config=generation_config)

    generated_table = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0].strip()

    inputs = requests.post(endpoint, headers=headers, json=data)

    prompt = build_prompt(TemplateKey.QA)

    response = gpt4(prompt, temperature, top_p, max_tokens=2000)

    if response.status_code == 200:
        response = response.json()
        return response["completions"][0]["content"]
    else:
        return "Error"