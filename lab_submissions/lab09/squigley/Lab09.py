from sklearn.naive_bayes import MultinomialNB, BernoulliNB  # Naive Bayes model classes
from sklearn.feature_extraction.text import CountVectorizer

import wikipedia as wiki  # wikipedia api wrapper
wiki.set_rate_limiting(True)  # might actually speed things up.
from sklearn.cross_validation import train_test_split  # split the data you have into training and test sets

def fetch_wiki(title, lang):
    '''
    Return the regular English or simple versions of an article.
    Simple versions are far shorter than the regular ones, so only
    pull the summary of regular articles.
    In case of an error, just return None instead of crashing the program.
    '''
    assert lang in ('en', 'simple'), "Language must be 'en' or 'simple'"

    try:
        wiki.set_lang(lang)
        page = wiki.page(title)
        # print page.title  # used for testing the function
        return (page.summary, 1) if lang == 'en' else (page.content, 0)  # 1: english, 0: simple
    except:  # NOTE: you should never have a blind `except` like this. but, hey, we're hacking.
        print ' - error with ' + lang + ' page for: ' + title
        return None

articles = ['General Relativity', 'Bayes Theorem', 'Ada Lovelace',
            'jackfruit', 'mantis shrimp', 'Der Ring des Nibelungen',
            'antikythera mechanism', 'teflon', 'superconductor',
            'Harper Lee', 'durian', 'Shostakovich', 'steel',
            'database', 'transistor', 'Goethe', 'dog', 'meme', 'spleen',
            'morphine', 'maple', "rubik's cube", 'souffle', 'chlorine',
            'earthworm', 'prune', 'ballet', 'ultrasound', 'bruce lee']

corpus = []

# search for each article, and if we get a result, store it in the corpus
for article in articles:
    en = fetch_wiki(article, 'en')
    if en:
        corpus.append(en)
    simple = fetch_wiki(article, 'simple')
    if simple:
        corpus.append(simple)
        
# convert features
text, Y = zip(*corpus)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text)

# Use SKLearn's train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(X, Y)

# Create our classifier
clf = MultinomialNB().fit(xtrain, ytrain)

print "Accuracy: %0.2f%%" % (100 * clf.score(xtest, ytest))

clf = BernoulliNB().fit(xtrain, ytrain)

print "Accuracy: %0.2f%%" % (100 * clf.score(xtest, ytest))


