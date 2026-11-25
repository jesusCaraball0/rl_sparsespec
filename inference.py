from transformers import AutoTokenizer
from serve.request.request import Request
from serve.run import run_inference

MODEL = "models/Qwen3-1.7B"
PROMPT = "What is 17 * 23? Reply with just the number."
MAX_NEW_TOKENS = 64
SPEC_STRIDE = 6

tokenizer = AutoTokenizer.from_pretrained(MODEL)
token_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
)

outputs = run_inference(
    reqs=[Request(token_ids=token_ids, desired_output_length=MAX_NEW_TOKENS)],
    model_name=MODEL,
    tp_ranks=1,
    max_batch_size=1,
    kv_cache="pillar",
    admit_policy="oracle",
    spec_stride=SPEC_STRIDE,
    budget_ratio=0.05,
    num_min_budget=128,
    temperature=0.0,
    dtype="bfloat16",
    max_mem_per_rank=8,
    clip_length=len(token_ids) + MAX_NEW_TOKENS + SPEC_STRIDE + 1,
)
print(outputs[0]["generated_tokens"])
