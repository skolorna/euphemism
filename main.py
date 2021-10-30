"""
Hello World
"""
import spacy
nlp = spacy.load("sv_pipeline")

# with open("sample.txt", encoding="utf8") as sample:
#     with open("output.txt", mode="w", encoding="utf8") as file:
#         for line in sample:
#             doc = nlp(line.rstrip())
#             for token in doc:
#                 file.write(f"{token.text} {token.pos_} {token.dep_}\n")

#             file.write("\n")

def rewrite(meal: str):
    doc = nlp(meal)

    for token in doc:
        if token.pos_ == "NOUN":
            print("got noun!")
        print(token.text, token.pos_, token.dep_)

rewrite("Köttbullar, stuvade makaroner och grönsaker")
rewrite("Köttbullar med makaroner och ketchup")
