import fitz

doc = fitz.open("repealedfileopen.pdf")

full_text = ""

for page in doc:
    full_text += page.get_text()

with open("ipc.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("Converted to ipc.txt")