"""
prepare_dataset.py
------------------
Transformă adnotare_finala.csv în formatul de antrenare pentru BERT.
Owner: x
Input:  data/processed/adnotare_finala.csv
Output: data/processed/dataset_train.csv, dataset_val.csv, dataset_test.csv
TODO:
- [ ] Citire adnotare_finala.csv și validare că toate aspectele sunt din lista fixă
- [ ] Conversie la vector de label-uri: pentru fiecare din cele 10 aspecte → pozitiv/negativ/neutru/nemenționat
- [ ] Adăugare coloană overall_sentiment
- [ ] Split stratificat 80/10/10 păstrând distribuția de clase
- [ ] Verificare că nu există recenzii duplicate între train/val/test


"""