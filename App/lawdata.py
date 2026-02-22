import requests
import json

print("📥 Fetching raw JSON files from Hugging Face...")

# Direct links to the raw JSON files
file_urls = [
    "https://huggingface.co/datasets/Techmaestro369/indian-legal-texts-finetuning/resolve/main/constitution_qa.json",
    "https://huggingface.co/datasets/Techmaestro369/indian-legal-texts-finetuning/resolve/main/crpc_qa.json",
    "https://huggingface.co/datasets/Techmaestro369/indian-legal-texts-finetuning/resolve/main/ipc_qa.json"
]

def format_qa_pair(question, answer):
    return f"""Below is an instruction that describes a task.

### Instruction:
{question}

### Response:
{answer}
<|endoftext|>
"""

formatted_text = ""
total_pairs = 0

# Loop through each URL, download the JSON, and extract just the Q&A
for url in file_urls:
    filename = url.split('/')[-1]
    print(f"   Downloading and processing {filename}...")
    
    response = requests.get(url)
    data = response.json()
    
    for item in data:
        # We use .get() so it safely ignores any extra columns like 'id'
        q = item.get('question', '').strip()
        a = item.get('answer', '').strip()
        
        if q and a:
            formatted_text += format_qa_pair(q, a)
            total_pairs += 1

# Save the perfectly formatted text
file_name = "indian_law_data.txt"
with open(file_name, "w", encoding="utf-8") as f:
    f.write(formatted_text)

print(f"\n🎉 Success! Extracted {total_pairs} legal Question/Answer pairs.")
print(f"Data saved to '{file_name}'.")
print(f"Total characters in dataset: {len(formatted_text)}")