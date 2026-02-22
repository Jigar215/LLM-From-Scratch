import re

# Load IPC text
with open("ipc.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split into sections
sections = re.split(r"Section\s+\d+\.", text)

dataset = []

for sec in sections:
    sec = sec.strip()
    if len(sec) > 100:   # avoid empty / small noise
        title = sec.split("\n")[0][:80]

        sample = f"""### Instruction:
Explain this Indian Penal Code law:

### Input:
{title}

### Response:
{sec}
"""
        dataset.append(sample)

# Save dataset
with open("law_dataset.txt", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(item + "\n")

print("Dataset created successfully!")