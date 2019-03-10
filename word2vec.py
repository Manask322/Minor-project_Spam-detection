import math
import sys
import gensim
import warnings
import re
from itertools import chain
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
 
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import wordnet
from nltk.corpus import stopwords

def main():
	
	dataset=load_file() 
	dataset=get_vectors(dataset)

	# k=dataset[0]
	# print(k.label,k.words,len(k.vectors),k.vectors[0])


def get_vectors(dataset):

	'''the sentences obtained from dataset are sent into word2vec 
	to obtain word wise vectors for each sentence'''
	
	total_sentences=len(dataset)
	sentences=[temp.words for temp in dataset]
	
	model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=600000)
	c=0
	k=0
	for sentence in range(total_sentences):
		for word in dataset[sentence].words:
			k+=1
			synonyms = wordnet.synsets(word)
			wordnet_synonyms=list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
			
			if word in model.vocab:
				#if word exists in word2vec
				dataset[sentence].vectors.append(model[word])

			elif len(wordnet_synonyms)!=0:
				'''if word does not exist in word2vec, find synonym using wordnet'''
				
				for i in wordnet_synonyms:
					if i in model.vocab:
						#use vector of similar word from wordnet
						dataset[sentence].vectors.append(model[i])
						break
			else:
				#need to use ConceptNet
				c+=1
				#print(word)

	print('\nun-identified words:',c,'\ntotal words:',k)
	return dataset


def load_file():
	'''loads the datafile and stores label,text attributes to an SMS_datat object'''
	file_names=['datasets/smsspamcollection/SMSSpamCollection']
	file=0

	dataset=[]

	with open(file_names[file]) as f:
		for line in f:
			words=line.split()

			temp=SMS_data()
			temp.label=words[0] #accessing the label of text (spam or ham)
			temp.words=text_processing(' '.join(words[1:])) #obtaining the text as list of words
			
			dataset.append(temp)

	return dataset

def text_processing(text):

	words=re.sub('[^A-Za-z0-9]+', ' ', text) #removes special characters
	#words=re.split('(\d+)',words)
	stop = set(stopwords.words('english'))
	words=[i for i in words.lower().split() if i not in stop] #stop word removal

	return words

class SMS_data:
	def __init__(self):
		self.label=None
		self.words=None
		self.vectors=[]

if __name__ == '__main__':
	main()