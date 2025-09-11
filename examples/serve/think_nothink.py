from transformers import AutoTokenizer
from tensorrt_llm import LLM, SamplingParams

def show_prompts(model_id: str, messages, llm):
    print(f"\n===== {model_id} =====")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(type(tok))
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    def apply(enable: bool):
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable,
        )

    # Try printing with enable_thinking toggled
    try:
        off_prompt = apply(False)
        on_prompt = apply(True)

        print("\n--- enable_thinking=False ---")
        print(off_prompt)
        print("\n--- Generated with LLM ---")
        print(llm.generate([off_prompt], sampling_params=SamplingParams(max_tokens=1024))[0].outputs[0].text)
        print("\n--- enable_thinking=True ---")
        print(on_prompt)
        print("\n--- Generated with LLM ---")
        print(llm.generate([on_prompt], sampling_params=SamplingParams(max_tokens=1024))[0].outputs[0].text)
    except TypeError:
        # Some tokenizers may not accept enable_thinking
        default_prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("\n--- enable_thinking not supported; default prompt ---")
        print(default_prompt)

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        # {"role": "user", "content": "What is 17 * 19? Show your reasoning briefly. /think"},
        {"role": "user", "content": "What is 17 * 19? Show your reasoning briefly. /no_think"},
    ]

    llm = LLM(model="/raid-models/Qwen3-14B/", trust_remote_code=True)

    # DeepSeek R1 Distill (uses thinking tags)
    show_prompts("deepseek-ai/DeepSeek-R1", messages, llm)

    # Qwen3 8B Instruct
    show_prompts("Qwen/Qwen3-8B", messages, llm)