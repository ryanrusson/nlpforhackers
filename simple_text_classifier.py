from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

news = fetch_20newsgroups(subset='all')

print(len(news.data))
print(len(news.target_names))
print(news.target_names)

for text, num_label in zip(news.data[:10], news.target[:10]):
    print('[%s]:\t\t "%s ..."' % (news.target_names[num_label], text[:100].split('\n')[0]))


def train(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    classifier.fit(X_train, y_train)
    print("Accuracy: %s" % classifier.score(X_test, y_test))
    return classifier


trial1 = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB()),
])

train(trial1, news.data, news.target)
