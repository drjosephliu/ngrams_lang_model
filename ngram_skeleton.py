import math, random
from collections import defaultdict

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
            model.update(line.strip())
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
        self.vocab = set()

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        all_ngrams = ngrams(self.n, text)
        for ngram in all_ngrams:
            self.ngram_counts[ngram] += 1
            self.context_counts[ngram[0]] += 1
            self.vocab.add(ngram[1])
        

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        if context not in self.context_counts:
            return 1 / len(self.vocab)
        if self.context_counts[context] == 0: 
            return 0
        return self.ngram_counts[(context, char)] / self.context_counts[context]

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
        if self.n == 0:
            return ""
        context = start_pad(self.n)
        random_text = ""
        for i in range(length):
            random_text += random_char(context)
            context = random_text[-self.n:]
        return random_text


    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        pass

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        pass

    def get_vocab(self):
        pass

    def update(self, text):
        pass

    def prob(self, context, char):
        pass

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    pass
