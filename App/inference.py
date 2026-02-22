import torch
import tiktoken
from model import GPTModel

# -------- DEVICE --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- TOKENIZER --------
tokenizer = tiktoken.get_encoding("gpt2")

# -------- CONFIG --------
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 1024,
    "n_layers": 24,
    "n_heads": 16
}

# -------- LOAD MODEL --------
model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load("gpt2-medium355Mlaw-sft.pth", map_location=device))
model.to(device)
model.eval()

# -------- HELPERS --------
def text_to_token_ids(text, tokenizer):
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze().tolist())

def format_chat_prompt(user_input):
    return f"""Below is an instruction that describes a task.

### Instruction:
{user_input}

### Response:
"""

# -------- GENERATION --------
def generate(model, idx, max_new_tokens, context_size, eos_id=None):

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

        if eos_id is not None and idx_next.item() == eos_id:
            break

    return idx

def generate_response(user_input):

    prompt = format_chat_prompt(user_input)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=200,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )

    generated_text = token_ids_to_text(token_ids, tokenizer)
    response = generated_text[len(prompt):].strip()

    return response

# -------- MAIN RUN --------
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = generate_response(user_input)
        print("Bot:", response)