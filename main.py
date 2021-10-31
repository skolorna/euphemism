from typing import Set
import spacy
nlp = spacy.load("sv_pipeline")


def tokenize(meal: str) -> str:
    doc = nlp(meal)
    out = set()

    for token in doc:
        if token.tag == 92:
            print(token.text, token.head.text)
            out.add(token.head.norm_)

    return " ".join(out)


def shingle(val: str):
    shingles = set()

    for i in range(len(val) - 1):
        shingles.add(val[i: i + 2])

    return shingles


def jaccard_index(a: Set, b: Set) -> float:
    return len(a.intersection(b)) / len(a.union(b))

# def similarity(a: list[str], b: list[str]) -> float:


if __name__ == "__main__":
    # a = shingle(tokenize("BBQ-kryddad kyckling, chilimajonnäs, bulgur, tortilla och rostad majs"))
    # b = shingle(tokenize("BBQ-kryddad beanit (ärtprotein), chilimajonnäs, bulgur, tortilla och rostad majs"))
    # c = shingle(tokenize("BBQ-kryddade sojabitar, chilimajonnäs, bulgur, tortilla och rostad majs"))

    groups = []

    # print("AB", jaccard_index(a, b))
    # print("AC", jaccard_index(a, c))
    # print("BC", jaccard_index(b, c))

    with open("smol.txt", encoding="utf8") as file:
        for line in file:
            tokens = tokenize(line.rstrip())
            a = shingle(tokens)

            for (b, text) in groups:
                if jaccard_index(a, b) > 0.6:
                    break
            else:
                groups.append((a, tokens))

    for (shingles, text) in groups:
        print(text)
