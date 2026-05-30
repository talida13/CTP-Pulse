
"""
train.py
--------
Fine-tuning bert-base-romanian-cased pe task-ul ABSA cu cele 10 aspecte fixe.
Owner: 
Input:  data/processed/dataset_train.csv, dataset_val.csv + model de bază de pe HuggingFace
Output: model/ctp_absa_bert/ (weights + tokenizer + config)
TODO:
- [ ] Tokenizare text cu bert-base-romanian-cased tokenizer (max 512 tokens)
- [ ] 10 capete de clasificare x 4 clase (pozitiv/negativ/neutru/nemenționat) + 1 cap overall
- [ ] Training loop cu HuggingFace Trainer, loss = cross-entropy sumată per cap
- [ ] Early stopping pe F1 de validare
- [ ] Salvare model cu save_pretrained() la finalul antrenării
- [ ] Rulare pe Google Colab cu GPU T4


"""