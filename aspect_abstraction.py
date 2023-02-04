import spacy
import pandas as pd




"""Define a function to extract keywords"""
def get_aspects(x):
    """Create a list of common words to remove"""
    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                  "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                  "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                  "these",
                  "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
                  "do",
                  "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
                  "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
                  "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
                  "again",
                  "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
                  "each",
                  "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
                  "than",
                  "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "reason", "condition", "kids"]

    """Load the pre-trained NLP model in spacy"""
    nlp = spacy.load("en_core_web_lg")

    doc=nlp(x) ## Tokenize and extract grammatical components
    doc=[i.text for i in doc if i.text not in stop_words and i.pos_=="NOUN"] ## Remove common words and retain only nouns
    doc=list(map(lambda i: i.lower(),doc)) ## Normalize text to lower case
    doc=pd.Series(doc)
    doc=doc.value_counts().head().index.tolist() ## Get 5 most frequent nouns
    return doc



if __name__ == "__main__":
    text = "The food was lousy - too sweet or too salty and the portions tiny."
   # get_aspects(text)
    print(get_aspects(text))
