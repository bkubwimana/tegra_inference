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

# Base prompt template with dynamic elements for experiments
BASE_PROMPT_TEMPLATE = """You are a constrained response language model.
Your task is to answer the following query with minimal processing.
Your output must strictly follow this format without any extra tokens or commentary.

Required Format:
BEGIN_RESPONSE
[Your Answer: up to {max_tokens} tokens]
END_RESPONSE

INPUT:
USER_QUERY: {user_query}
OUTPUT_RANGE: up to {max_tokens} tokens
PRIORITY: {priority}
TASK_COMPLEXITY: {task_complexity}
RESPONSE_DETAIL: {response_detail}
TIME_BUDGET: {time_budget}
"""

class PromptEngine:
    def __init__(self):
        self.params = {
            "max_tokens": 50,
            "user_query": "Analyze the scene for object detection.",
            "priority": "quality",
            "task_complexity": "minimal",
            "response_detail": "concise",
            "time_budget": "ultra-fast"
        }
        self.base_prompt_template = BASE_PROMPT_TEMPLATE
        model_path = "deepseek-ai/Janus-Pro-7B"
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    def set_prompt_params(self, params: dict):
        """Update dynamic parameters for prompt formatting."""
        self.params.update(params)

    def get_response(self, query: str, image_path: str = None) -> tuple:
        """
        Uses the current parameters (with 'user_query' overridden by query) to build the prompt,
        then performs inference to get a response.
        If image_path is provided, it is passed as a multimodal input.
        
        Returns:
            A tuple (answer, total_latency, generation_latency, decode_latency)
        """
        local_params = self.params.copy()
        local_params["user_query"] = query
        # print(f"\033[91mUsing prompt parameters: {local_params}\033[0m")

        formatted_prompt = self.base_prompt_template.format(**local_params)

        if image_path:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{formatted_prompt}",
                    "images": [image_path],
                },
                {"role": "<|Assistant|>", "content": ""}
            ]
        else:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": formatted_prompt,
                },
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
        #calculate token count from the answer
        tokens = self.tokenizer.tokenize(answer)
        token_count = len(tokens)
        return formatted_prompt, answer, total_latency, generation_latency, decode_latency, token_count