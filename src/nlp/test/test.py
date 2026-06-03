import json
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("model/ctp_aspect_pair_detector")

texts = [
    "Autobuzul a întârziat 30 de minute și era foarte aglomerat.",
    "Autobuzul a venit la timp, nu era aglomerat și am găsit loc pe scaun.",
    "POS-ul din tramvai nu merge și nu am putut cumpăra bilet.",
    "POS-ul din tramvai a funcționat corect și am putut plăti rapid cu cardul.",
    "Stația este curată, are panou electronic și suficient spațiu pentru călători.",
    "Controlorii au fost agresivi și au vorbit urât cu pasagerii.",
    "Controlorii au fost civilizați și au verificat biletele fără comentarii inutile.",
    "Aplicația arăta greșit timpul de sosire al autobuzului.",
    "Autobuzul era murdar și mirosea urât.",
    "Șoferul a condus prudent și a așteptat ca pasagerii să urce.",
]

meta = json.loads((MODEL_DIR / "aspect_pair_config.json").read_text(encoding="utf-8"))
aspects = meta["aspects"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

THRESHOLD = 0.80

def make_input(aspect: str, text: str) -> str:
    return f"aspect: {aspect}. text: {text}"

for text in texts:
    rows = []

    for aspect in aspects:
        pair_text = make_input(aspect, text)

        inputs = tokenizer(
            pair_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0]
        present_prob = float(probs[1])

        rows.append((aspect, present_prob))

    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    detected = [(a, p) for a, p in rows if p >= THRESHOLD]

    print("\n" + "=" * 90)
    print("TEXT:")
    print(text)

    print("\nASPECTE DETECTATE:")
    if detected:
        for aspect, prob in detected:
            print(f"- {aspect}: {prob:.3f}")
    else:
        print("- niciun aspect peste prag")

    print("\nTOP 5 CANDIDAȚI:")
    for aspect, prob in rows[:5]:
        print(f"- {aspect}: {prob:.3f}")
