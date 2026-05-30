"""
sentiment_classifier.py
-----------------------
Clasificare polaritate per aspect: POZITIV / NEGATIV / NEUTRU.
Implementează atât baseline lexical cât și modelul BERT-ro.

Owner: Postolache Andrei
Input: (text, aspect_code)
Output: {'label': 'NEG'|'POS'|'NEU', 'score': float}

TODO:
- [ ] Baseline: SentiWordNet adaptat pentru română
- [ ] Model: fine-tuning readerbench/bert-base-romanian
- [ ] Wrapper unificat care expune ambele modele
- [ ] Batch inference pentru dataset complet
"""
