import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

class VikhrLLM:
    def __init__(self, model_name="Vikhrmodels/Vikhr-7B-instruct_0.4", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        self.device = device
        if device == "cuda":
            self.model = self.model.to(device)
        self.model.eval()

    async def complete(self, prompt: str, system_prompt: str = None, max_new_tokens: int = 512) -> str:
        if system_prompt is None:
            system_prompt = "Ты — Вихрь, русскоязычный автоматический ассистент. Отвечай на русском языке, используя только предоставленный контекст."
        full_prompt = f"<s>system\n{system_prompt}</s>\n<s>user\n{prompt}</s>\n<s>bot\n"
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, generation_config=generation_config)
        response = self.tokenizer.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return response.strip()