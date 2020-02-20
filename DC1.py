import collections
import itertools
import os,glob, re
import nltk.classify.util, nltk.metrics

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import word_tokenize

'''text = "God is Great! I won a lottery."
filtered_list = [word for word in word_tokenize(text) if word.lower() not in stopwords.words('english')]
print(word_tokenize(text))
print(filtered_list)'''

pathunlabled = "/Users/yanyan/PycharmProjects/NLP-DC1/unlabeled"
pathpositive = "/Users/yanyan/PycharmProjects/NLP-DC1/pos"
pathnegative = "/Users/yanyan/PycharmProjects/NLP-DC1/neg"

path = "/Users/yanyan/PycharmProjects/NLP-DC1/unlabeled"
direct = "/Users/yanyan/PycharmProjects/NLP-DC1/ratings"


desired_rate = []
input_reviews = []
flags=[]

rates = open(os.path.join(direct, "unlabeled.txt"), "r")
review_rate = rates.readlines()
for rate in review_rate:
    desired_rate.append(float(rate.split()[1]))
#print(desired_rate)
rates.close()

def tokened_list(path):
    input_reviews = []
    for f in os.listdir(path):
        if f[-4:] == '.txt':
            file = open(path + "/" + f, 'r')
            input_reviews.append(
                [word for word in word_tokenize(file.read()) if word.lower() not in stopwords.words('english')])
    return input_reviews


# bigram collocation to increase accuracy
# not sure if I am using this correctly -Alexis
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

# print(tokened_list(pathpositive)[1])



# sort the filename with both alphabet and numbers
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

for f in sorted(os.listdir(path),key=natural_key):
    if f[-4:] == '.txt':
        print(f)
        file = open(path + "/" + f, 'r')
        input_reviews.append(file.read())




def features(list):
    # filtered_list = [word for word in list if word.lower() not in stopwords.words('english')]
    return dict([(word, True) for word in list])


if __name__ == '__main__':
    # Load positive and negative reviews
    positive_fileid = movie_reviews.fileids('pos')
    negative_fileid = movie_reviews.fileids('neg')

features_positive = [(features(f), 'Positive') for f in tokened_list(pathpositive)]
features_negative = [(features(f), 'Negative') for f in tokened_list(pathnegative)]

# Split the data into train and test (8/2)
threshold_factor = 0.8
threshold_positive = int(threshold_factor * len(features_positive))
threshold_negative = int(threshold_factor * len(features_negative))

features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]



classifier = NaiveBayesClassifier.train(features_train)
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

#print(refsets)

for i, (feats, label) in enumerate(features_test):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)
'''
# if you comment these print lines out below, you will see the improved accuracy
print('accuracy:', nltk.classify.util.accuracy(classifier, features_test))
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
classifier.show_most_informative_features()
'''

print("Accuracy of the classifier: ", nltk.classify.util.accuracy(classifier, features_test))

#print("Predictions: ")
#Add flags for inconsistant reviews

#for review in input_reviews[:10]:
for x in range(0, 10):
#for x in range(0, 10len(input_reviews)):
    review = input_reviews[x]
    print("\nReview:", review)
    probdist_sentiment = classifier.prob_classify(features(review.split())).max()

    if (probdist_sentiment =="Positive" and desired_rate[x] < 3.0) or (probdist_sentiment =="Negative" and desired_rate[x] > 3.0):
        print("fake review")
        flags.append(review)
    else:
        print("good")


    #print("Predicted sentiment: ", pred_sentiment)
    #print("Probability: ", round(probdist.prob(pred_sentiment), 2))




'''
#tokenized the reviews
positive_tokened = [movie_reviews.words(fileids=[f])for f in positive_fileid]
negative_tokened = [movie_reviews.words(fileids=[f])for f in negative_fileid]

#filterout the stopwords
filtered_words_pos = [word for word in positive_tokened if word not in stopwords.words('english')]
filtered_words_neg = [word for word in negative_tokened if word not in stopwords.words('english')]

#stop_words = list(get_stop_words('en'))         #About 900 stopwords
#nltk_words = list(stopwords.words('english')) #About 150 stopwords
#stop_words.extend(nltk_words)
#output = [w for w in word_list if not w in stop_words]
features_positive = [(features(f), 'Positive') for f in filtered_words_pos]
features_negative = [(features(f), 'Negative') for f in filtered_words_neg]
'''