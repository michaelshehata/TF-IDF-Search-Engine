import index
import pickle
import matplotlib.pyplot as plt
import math
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter

PRECISION = 10  # Precision@k value
QUERYEXPANSION = True  # Flag to enable or disable query expansion

def load_from_pickle():
    with open("vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    with open("docids.pkl", 'rb') as f:
        docIDs = pickle.load(f)
    with open("postings.pkl", 'rb') as f:
        postings = pickle.load(f)
    return vocab, docIDs, postings

def is_relevant(doc_id, query):
    """
    Check if a document is relevant to the query.
    Customize this function based on your actual relevance criteria.
    """
    return query.lower() in docIDs[doc_id]["contents"].lower()

def expand_query(query):
    """
    Expand the query with synonyms of nouns using WordNet.
    """
    query_tokens = word_tokenize(query)
    pos_tags = pos_tag(query_tokens)

    # Only expand nouns (NN, NNS, etc.)
    synonyms = []
    for word, tag in pos_tags:
        if tag.startswith("NN"):
            synsets = wordnet.synsets(word)
            for synset in synsets:
                for lemma in synset.lemmas():
                    if lemma.name() != word:
                        synonyms.append(lemma.name())
    return Counter(synonyms)

def retrieve_top_k(query, k=10):
    query_tokens = index.preprocess(query)

    if QUERYEXPANSION:
        expanded_terms = expand_query(query)
        for term in expanded_terms:
            query_tokens[term] = query_tokens.get(term, 0) + 0.4  # Add expanded terms with reduced weight

    query_tfidf = [
        calculate_query_tfidf(query_tokens, term_id, term, postings, len(docIDs))
        for term, term_id in vocab.items()
    ]

    document_tfidf = [
        [
            calculate_document_tfidf(doc_id, term_id, postings, len(docIDs))
            for term_id in postings.keys()
        ]
        for doc_id in range(len(docIDs))
    ]

    similarity_scores = {
        doc_id: cosine_similarity(query_tfidf, document_tfidf[doc_id])
        for doc_id in range(len(docIDs))
    }

    ranked_documents = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_documents[:k]

def calculate_precision_at_k(query, ranked_docs, k=10):
    relevant_count = sum(1 for doc in ranked_docs[:k] if is_relevant(doc[0], query))
    return relevant_count / k

def compare_retrieval_methods(queries, results_with_expansion, results_without_expansion, k=10):
    precision_with_expansion = [
        calculate_precision_at_k(query, ranked_docs, k)
        for query, ranked_docs in zip(queries, results_with_expansion)
    ]

    precision_without_expansion = [
        calculate_precision_at_k(query, ranked_docs, k)
        for query, ranked_docs in zip(queries, results_without_expansion)
    ]

    x = range(len(queries))

    plt.figure(figsize=(10, 6))
    plt.plot(x, precision_with_expansion, label='With Query Expansion', marker='o', linestyle='-', color='blue')
    plt.plot(x, precision_without_expansion, label='Without Query Expansion', marker='s', linestyle='--', color='red')
    plt.xticks(x, queries, rotation=45, ha='right')
    plt.xlabel('Queries')
    plt.ylabel(f'Precision@{k}')
    plt.title('Comparison of Retrieval Methods')
    plt.legend()
    plt.tight_layout()
    plt.show()

def cosine_similarity(query_tfidf, document_tfidf):
    num = sum(q * d for q, d in zip(query_tfidf, document_tfidf))
    denom = (sum(x**2 for x in query_tfidf)**0.5) * (sum(x**2 for x in document_tfidf)**0.5)
    return num / denom if denom != 0 else 0

def calculate_query_tfidf(tokens, term_id, term, postings, N):
    tf = tokens.get(term, 0)
    df = len(postings.get(term_id, {}))  # Number of documents containing the term
    return calculate_tf_idf(tf, df, N)

def calculate_document_tfidf(doc_id, term_id, postings, N):
    tfTuple = postings[term_id].get(doc_id, (0, 1))  # Default frequency 0, weight 1
    tf = tfTuple[0]
    term_weight = tfTuple[1]
    df = len(postings[term_id])
    return calculate_tf_idf(tf, df, N, term_weight)

def calculate_tf_idf(tf, df, N, term_weight=1.0):
    return ((term_weight * (1 + math.log(tf))) * math.log(N/df)) if tf > 0 and df > 0 else 0

vocab, docIDs, postings = load_from_pickle()

if __name__ == "__main__":
     queries = ["Devil Kings","Dynasty Warriors","Sports Genre Games","Hunting Genre Games", "Game Developed by Eurocom",'Game Published by Activision',"Game Published by Sony Computer Entertainment","Teen PS2 Games"]

    # Precision with query expansion
    QUERYEXPANSION = True
    results_with_expansion = [retrieve_top_k(query, PRECISION) for query in queries]

    # Precision without query expansion
    QUERYEXPANSION = False
    results_without_expansion = [retrieve_top_k(query, PRECISION) for query in queries]

    compare_retrieval_methods(queries, results_with_expansion, results_without_expansion, PRECISION)
