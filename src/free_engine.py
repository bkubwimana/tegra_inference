import os
import time
import torch
from transformers import AutoModelForCausalLM
janus_path = os.path.join(os.path.dirname(__file__), "..")
import sys
sys.path.append(janus_path)
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from latency_manager import record_latency

class FreePromptEngine:
    def __init__(self):
        model_path = "deepseek-ai/Janus-Pro-7B"
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    def get_response(self, query: str, image_path: str = None) -> tuple:
        """
        For free VQA tests, the prompt simply consists of the raw query.
        Returns:
            A tuple (original_query, answer, total_latency, generation_latency, decode_latency, token_count)
        """
        # Build a simple conversation with just the query.
        if image_path:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{query}",
                    "images": [image_path],
                },
                {"role": "<|Assistant|>", "content": ""}
            ]
        else:
            conversation = [
                {"role": "<|User|>", "content": query},
                {"role": "<|Assistant|>", "content": ""}
            ]
        
        pil_images = load_pil_images(conversation) if image_path else []
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)
        
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        
        with record_latency() as gen_latency:
            outputs = self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )
        generation_latency = gen_latency.elapsed
        
        decode_start = time.time()
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        decode_latency = time.time() - decode_start
        total_latency = generation_latency + decode_latency
        token_count = len(self.tokenizer.tokenize(answer))
        
        return query, answer, total_latency, generation_latency, decode_latency, token_count