"""
evaluate.py
-----------
Calculează metricile modelului pe setul de test și compară metodele.
Owner: 
Input:  data/processed/dataset_test.csv + model/ctp_absa_bert/
Output: afișare tabel în terminal + data/processed/evaluation_results.csv
TODO:
- [ ] Rulare predicții pe tot dataset_test.csv
- [ ] Calcul Precision, Recall, F1 per aspect pentru modelul BERT
- [ ] Același calcul pentru baseline SentiWordNet (pentru comparație)
- [ ] Tabel comparativ: SentiWordNet vs BERT fine-tuned, per aspect și overall
- [ ] Matrice de confuzie per aspect
- [ ] Salvare rezultate în evaluation_results.csv
"""