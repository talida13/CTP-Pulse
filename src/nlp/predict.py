"""
predict.py
----------
Încarcă modelul antrenat și face predicție ABSA pe un review nou.
Owner: 
Input:  text review (string) + model/ctp_absa_bert/
Output: dict cu aspects: [{aspect, sentiment, score}] + overall_sentiment + overall_score
TODO:
- [ ] Încărcare model și tokenizer din model/ctp_absa_bert/
- [ ] Tokenizare și inferență pe textul primit
- [ ] Conversie logits → probabilități → label per aspect
- [ ] Filtrare aspecte cu label nemenționat (nu le include în output)
- [ ] Calcul overall_sentiment ca medie ponderată a scorurilor individuale
"""