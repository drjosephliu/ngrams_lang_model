import math, random, os
from collections import defaultdict
# from nltk.util import ngrams

################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    ngrams = []
    padded = start_pad(n) + text
    for i in range(n, len(padded)):
        ngrams.append((padded[i-n:i], padded[i]))
    return ngrams

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip()+'.')
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = defaultdict(int)

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return set(self.vocab.keys())

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        all_ngrams = ngrams(self.n, text)
        for ngram in all_ngrams:
            self.ngram_counts[ngram] += 1
            self.context_counts[ngram[0]] += 1
            self.vocab[ngram[1]] += 1
        

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        if context not in self.context_counts:
            return 1 / len(self.vocab)
        if self.context_counts[context] == 0 and self.k == 0:
            return 0
        numer = self.ngram_counts[(context, char)] + self.k
        denom = self.context_counts[context] + self.k * len(self.vocab)
        return numer / denom

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        vs = sorted(self.vocab)
        r = random.random()
        cum_prob = 0
        for v in vs:
            cum_prob += self.prob(context, v)
            if cum_prob > r:
                return v
            

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        context = start_pad(self.n)
        random_text = ""
        for i in range(length):
            r_char = self.random_char(context)
            random_text += r_char
            context = context[1:] + r_char
        return random_text


    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        ngs = ngrams(self.n, text)
        cum_prob = float(0)
        for ng in ngs:
            prob = self.prob(ng[0], ng[1])
            if prob == 0:
                return float('inf')
            cum_prob += math.log(prob)
        return math.e ** (cum_prob * (-1 / len(text)))



################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = defaultdict(int)
        self.lamda = [1 / (n + 1)] * (n + 1)

    def get_vocab(self):
        return set(self.vocab.keys())

    def set_lamda(self, new_lamda):
        assert(len(self.lamda) == len(new_lamda) and round(sum(new_lamda)) == 1)
        self.lamda = new_lamda

    def update(self, text):
        for ch in text:
            self.vocab[ch] += 1
        for i in range(0, self.n+1):
            all_ngrams = ngrams(i, text)
            for ngram in all_ngrams:
                self.ngram_counts[ngram] += 1
                self.context_counts[ngram[0]] += 1


    def prob(self, context, char):
        cum_prob = 0
        for i in range(0, self.n+1):
            if context not in self.context_counts:
                cum_prob += self.lamda[i] * (1 / len(self.vocab))
                context = context[1:]
                continue
            numer = self.ngram_counts[(context, char)] + self.k
            denom = self.context_counts[context] + self.k * len(self.vocab)
            cum_prob += self.lamda[i] * (numer / denom)
            context = context[1:]
        return cum_prob

    # def kneser_ney_prob(self, context, char):


################################################################################
# Part 3: Your N-Gram Model Experimentation


class CityClassifier(object):

    def __init__(self):
        self.models = {}

    def train_models(self, model_class, n, k):
        for data_file in os.listdir('train'):
            path = 'train/' + data_file
            country = data_file[:2]
            m = create_ngram_model_lines(model_class, path, n, k)
            self.models[country] = m

    def perplexity(self, text):
        results = {}
        for country, model in self.models.items():
            perplexity = model.perplexity(text)
            results[country] = perplexity
        return results

#################################################################################

#####CUSTOM MODEL

##Load File
def load_test_file(data_file):
    words = []
    labels = []   
    with open(data_file, 'rt', encoding="ISO-8859-1") as f:
        for line in f:
            line_split = line[:-1].split("\t")
            words.append(line_split[0].lower())
                # labels.append(int(line_split[1]))
    return words

#Train
cc = CityClassifier()
n=4
k=.37
cc.train_models(NgramModelWithInterpolation, n, k)


#[0.05, 0.1, 0.2, 0.3, 0.35]
#[0.1, 0.15, 0.2, 0.25, 0.3]

#Interpolation
cc.models['af'].set_lamda([0.2, 0.2, 0.2, 0.2, 0.2])
cc.models['cn'].set_lamda([0.2, 0.2, 0.2, 0.2, 0.2])
cc.models['de'].set_lamda([0.2, 0.2, 0.2, 0.2, 0.2])
cc.models['fi'].set_lamda([0.2, 0.2, 0.2, 0.2, 0.2])
cc.models['fr'].set_lamda([0.2, 0.2, 0.2, 0.2, 0.2])
cc.models['in'].set_lamda([0.15, 0.15, 0.2, 0.25, 0.25])
cc.models['ir'].set_lamda([0.15, 0.15, 0.2, 0.25, 0.25])
cc.models['pk'].set_lamda([0.2, 0.2, 0.2, 0.2, 0.2])
cc.models['za'].set_lamda([0.2, 0.2, 0.2, 0.2, 0.2])


# for data_file in os.listdir('val'):
#     path = 'val/' + data_file
#     country = data_file[:2]
#     text = ""
#     with open(path, encoding='utf-8', errors="ignore") as f:
#         text = f.read()
#     pp = cc.perplexity(text)

#####TEST TRAIN
results = []
total_count = 0.0
total_len = 0
for data_file in os.listdir('train'):
    path = 'train/' + data_file
    country = data_file[:2]
    # print(country + "#####" + '\n')
    cities = load_test_file(path)
    count = 0.0
    for entry in cities:
        pp = cc.perplexity(entry+'.')
        # print("entry: {}, prediction: {}".format(entry, min(pp, key=pp.get))) 
        if min(pp, key=pp.get) == str(country):
            count+=1
    results.append(count/len(cities))
    total_count+=count
    total_len+=len(cities)
for i in range(len(results)):
    print("Train Country: {}, Percent Correct: {}".format(COUNTRY_CODES[i], results[i]))
print(total_count/total_len)

#####TEST VALIDATOIN
results = []
total_count = 0.0
total_len = 0
for data_file in os.listdir('val'):
    path = 'val/' + data_file
    country = data_file[:2]
    # print(country + "#####" + '\n')
    cities = load_test_file(path)
    count = 0.0
    for entry in cities:
        pp = cc.perplexity(entry+'.')
        # print("entry: {}, prediction: {}".format(entry, min(pp, key=pp.get))) 
        if min(pp, key=pp.get) == str(country):
            count+=1
    results.append(count/len(cities))
    total_count+=count
    total_len+=len(cities)
for i in range(len(results)):
    print("VAL Country: {}, Percent Correct: {}".format(COUNTRY_CODES[i], results[i]))
print(total_count/total_len)

results = []
test_cities = load_test_file('test/cities_test.txt')
outfile = open("test_labels.txt", "w")
for entry in test_cities:
    pp = cc.perplexity(entry+'.')
    results.append(min(pp, key=pp.get))
for s in results:
        outfile.write("%s\n" % s)
outfile.close()


if __name__ == '__main__':
    pass
    #COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']