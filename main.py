import spacy
nlp = spacy.load("sv_pipeline")

def summarize(meal: str) -> str:
    doc = nlp(meal)

    for token in doc:
        print(token.text, token.pos_, token.tag_)

if __name__ == "__main__":
    with open("sample.txt", encoding="utf8") as file:
        for line in file:
            summarize(line.rstrip())
