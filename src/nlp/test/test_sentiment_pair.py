import json
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("model/ctp_sentiment_pair")

examples = [
    {
        "aspect": "aglomeratie",
        "text": "Autobuzul a întârziat 30 de minute și era foarte aglomerat.",
        "fragment": "era foarte aglomerat",
    },
    {
        "aspect": "frecventa",
        "text": "Autobuzul a întârziat 30 de minute și era foarte aglomerat.",
        "fragment": "a întârziat 30 de minute",
    },
    {
        "aspect": "validare_bilete",
        "text": "POS-ul din tramvai nu merge și nu am putut cumpăra bilet.",
        "fragment": "POS-ul din tramvai nu merge",
    },
    {
        "aspect": "validare_bilete",
        "text": "POS-ul din tramvai a funcționat corect și am putut plăti rapid cu cardul.",
        "fragment": "POS-ul din tramvai a funcționat corect",
    },
    {
        "aspect": "curatenie",
        "text": "Autobuzul era murdar și mirosea urât.",
        "fragment": "era murdar și mirosea urât",
    },
    {
        "aspect": "controlori",
        "text": "Controlorii au fost agresivi și au vorbit urât cu pasagerii.",
        "fragment": "au fost agresivi și au vorbit urât",
    },
    {
        "aspect": "controlori",
        "text": "Controlorii au fost civilizați și au verificat biletele fără comentarii inutile.",
        "fragment": "au fost civilizați",
    },
    {
        "aspect": "siguranta",
        "text": "Șoferul a condus prudent și a așteptat ca pasagerii să urce.",
        "fragment": "a condus prudent și a așteptat ca pasagerii să urce",
    },
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def make_input(aspect, text, fragment):
    return f"aspect: {aspect}. fragment: {fragment}. text: {text}"

for ex in examples:
    input_text = make_input(ex["aspect"], ex["text"], ex["fragment"])

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs))
    pred_label = model.config.id2label[pred_id]

    print("\n" + "=" * 90)
    print("ASPECT:", ex["aspect"])
    print("TEXT:", ex["text"])
    print("FRAGMENT:", ex["fragment"])

    print("\nPROBABILITĂȚI:")
    for idx in range(len(probs)):
        label = model.config.id2label[idx]
        print(f"- {label}: {float(probs[idx]):.3f}")

    print("\nPREDICȚIE:")
    print(f"{pred_label} ({float(probs[pred_id]):.3f})")
