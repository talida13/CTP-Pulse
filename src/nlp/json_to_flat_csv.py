"""
jsonl_to_flat_absa_csv.py
-------------------------
Transformă adnotările ABSA generate de LLM în format JSONL
într-un CSV flat, potrivit pentru analiză și antrenare ulterioară.
Owner: Talida Caraman
Input:
    data/processed/reviews_merged.csv
    data/processed/annotations.jsonl
Output:
    data/processed/absa_flat_dataset.csv
Format input JSONL:
{"review_id":"...","aspects":[{"aspect":"...","sentiment":"...","fragment":"..."}],"overall_sentiment":"..."}
Format output CSV:
review_id,source,location,rating,text,aspect,sentiment,fragment,overall_sentiment
TODO:
[ ] Citire CSV original cu review-uri
[ ] Citire JSONL cu adnotări ABSA
[ ] Validare review_id și potrivire cu textul original
[ ] Flatten pentru lista de aspecte
[ ] Păstrare metadate: source, location, rating
[ ] Salvare CSV final pentru training/analiză
"""

import pandas as pd
import json

reviews_csv = "data/processed/reviews_merged.csv"
annotations_jsonl = "data/processed/reviews_annotated.jsonl"
output_csv = "data/processed/absa_flat_dataset.csv"

# CSV-ul original cu review-uri
reviews_df = pd.read_csv(reviews_csv)

# Adnotările produse de model
rows = []

with open(annotations_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)

        review_id = str(item.get("review_id"))
        aspects = item.get("aspects", [])
        overall_sentiment = item.get("overall_sentiment")

        # găsim review-ul original
        original = reviews_df[reviews_df["review_id"].astype(str) == review_id]

        if not original.empty:
            original_row = original.iloc[0]
            text = original_row.get("text", "")
            source = original_row.get("source", None)
            location = original_row.get("location", None)
            rating = original_row.get("rating", None)
        else:
            text = ""
            source = item.get("source")
            location = item.get("location")
            rating = item.get("rating")

        # dacă review-ul are aspecte
        for asp in aspects:
            rows.append({
                "review_id": review_id,
                "source": source,
                "location": location,
                "rating": rating,
                "text": text,
                "aspect": asp.get("aspect"),
                "sentiment": asp.get("sentiment"),
                "fragment": asp.get("fragment"),
                "overall_sentiment": overall_sentiment
            })

flat_df = pd.DataFrame(rows)
flat_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"Saved {output_csv}")
print(flat_df.head())