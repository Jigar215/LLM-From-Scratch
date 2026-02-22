import torch
import torch.nn.functional as F

def attention(query, keys, values, mask=None):
    d_k = keys.shape[-1]

    # Step 1: Dot Product
    scores = torch.matmul(query, keys.transpose(-2, -1)) / (d_k ** 0.5)

    # Step 2: Apply Mask (if any)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: Softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Step 4: Multiply with Values
    output = torch.matmul(attn_weights, values)

    return output, attn_weights


def run_attention():
    print("\nSimple Attention Demo Started!")
    print("Type something (or type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Exiting...")
            break

        # Convert sentence length into fake embeddings
        tokens = user_input.split()
        seq_len = len(tokens)

        if seq_len == 0:
            print("Bot: Please type something meaningful.")
            continue

        # Random embeddings for demo
        query = torch.rand(1, seq_len, 4)
        keys = torch.rand(1, seq_len, 4)
        values = torch.rand(1, seq_len, 4)

        output, attn_weights = attention(query, keys, values)

        print("\nAttention Weights:\n", attn_weights)
        print("\nOutput:\n", output)
        print("\nBot: Processed your input using attention!\n")


if __name__ == "__main__":
    run_attention()