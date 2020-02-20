import collections, os,re
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.metrics import precision,recall
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


# DC Medium
def evaluate_classifier(method):
    positive_fileid = movie_reviews.fileids('neg')
    negative_fileid = movie_reviews.fileids('pos')

    features_negative= [(method(movie_reviews.words(fileids=[f])), 'neg') for f in positive_fileid]
    features_positive = [(method(movie_reviews.words(fileids=[f])), 'pos') for f in negative_fileid]

    # Split the data into train and test (8/2)
    threshold_factor = 0.8
    threshold_positive = int(threshold_factor * len(features_positive))
    threshold_negative = int(threshold_factor * len(features_negative))

    trainfeats = features_positive[:threshold_positive] + features_negative[:threshold_negative]
    testfeats = features_positive[threshold_positive:] + features_negative[threshold_negative:]
    classifier = NaiveBayesClassifier.train(trainfeats)

    #create empty set to hold the features later on
    trainsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        trainsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
    print('pos precision:', precision(trainsets['pos'], testsets['pos']))
    print('pos recall:', recall(trainsets['pos'], testsets['pos']))
    print('neg precision:', precision(trainsets['neg'], testsets['neg']))
    print('neg recall:', recall(trainsets['neg'], testsets['neg']))
    #classifier.show_most_informative_features()
    return classifier


# Naive Baye
def features(words):
    return dict([(word, True) for word in words])




#With stop words
stopset = set(stopwords.words('english'))
def stopword(words):
    return dict([(word, True) for word in words if word.lower() not in stopset])


# Using the Bigram
def bigram(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

classifier_naive= evaluate_classifier(features)
classifier_stopword= evaluate_classifier(stopword)
classifier_bigram = evaluate_classifier(bigram)

'''------------------------------------------- DC Hard---------------------------------------------'''

input_reviews = []
desired_rate = []
flags_n=[]
flags_s=[]
flags_b=[]

#path = "/Users/yanyan/PycharmProjects/NLP-DC1/unlabeled"
#direct = "/Users/yanyan/PycharmProjects/NLP-DC1/ratings"
path = os.getcwd()+"/unlabeled"
direct = os.getcwd()+"/ratings"

# get the rate and convert to double
rates = open(os.path.join(direct, "unlabeled.txt"), "r")
review_rate = rates.readlines()
for rate in review_rate:
    desired_rate.append(float(rate.split()[1]))
rates.close()


# sort the filename with both alphabet and numbers
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# raise a flag
def raise_flag(probdist_sentiment,desired_rate,flags):
    if (probdist_sentiment == "pos" and desired_rate < 2.5) or (
        probdist_sentiment == "neg" and desired_rate > 2.5):
        flags.append(review)

for f in sorted(os.listdir(path),key=natural_key):
    if f[-4:] == '.txt':
        file = open(path + "/" + f, 'r')
        input_reviews.append(file.read())


#for x in range(0, 10):
for x in range(0, len(input_reviews)):
    review = input_reviews[x]
    rate = desired_rate[x]
    #print("\nReview:", review)
    probdist_naive = classifier_naive.prob_classify(features(review.split())).max()
    probdist_stopword = classifier_stopword.prob_classify(stopword(review.split())).max()
    probdist_bigram = classifier_bigram.prob_classify(bigram(review.split())).max()
    raise_flag(probdist_naive, rate,flags_n)
    raise_flag(probdist_stopword, rate,flags_s)
    raise_flag(probdist_bigram , rate,flags_b)
    #print("Predicted sentiment: ", probdist_sentiment)
    #print(desired_rate[x])
    #else:
    #    print("good")

print(len(flags_n))
print(len(flags_s))
print(len(flags_b))