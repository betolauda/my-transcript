#Train a Custom NER Component

import spacy



nlp_ner = spacy.load("path/to/output/model")
doc = nlp_ner("El Banco Central ajustó la tasa de interés.")
print([(ent.text, ent.label_) for ent in doc.ents])