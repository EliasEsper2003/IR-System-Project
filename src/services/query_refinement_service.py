from nltk.corpus import wordnet


def expand_query(query, max_synonyms=2):

    expanded_terms = []

    words = query.split()

    for word in words:

        expanded_terms.append(word)

        synonyms = set()

        for syn in wordnet.synsets(word):

            for lemma in syn.lemmas():

                synonym = lemma.name().replace("_", " ")

                if synonym.lower() != word.lower():
                    synonyms.add(synonym)

        synonyms = list(synonyms)[:max_synonyms]

        expanded_terms.extend(synonyms)

    return " ".join(expanded_terms)


def refine_query(query):

    return expand_query(query)