import math
import random
import sys
import gensim
import warnings
import re
from itertools import chain
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import numpy as np 


def main():
	
	dataset=load_file() 
	dataset=get_vectors(dataset)

	# k=dataset[0]
	# print(k.label,k.words,len(k.vectors),len(k.vectors[0]))


def get_vectors(dataset):

	'''the sentences obtained from dataset are sent into word2vec 
	to obtain word wise vectors for each sentence'''
	
	total_sentences=len(dataset)
	sentences=[temp.words for temp in dataset]
	
	#getting word2vec vectors
	model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=600000)
	
	
	#getting conceptnet vectors
	conceptnet_embeddings_index = {}
	with open('numberbatch-en-17.06.txt', encoding='utf-8') as f:
	    for line in f:
	        values = line.split(' ')
	        word = values[0]
	        embedding = np.asarray(values[1:], dtype='float32')
	        conceptnet_embeddings_index[word] = embedding
	c=0
	k=0
	w2v=0
	wn=0
	for sentence in range(total_sentences):
		for word in dataset[sentence].words:
			k+=1
			synonyms = wordnet.synsets(word)
			wordnet_synonyms=list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
			
			if word in model.vocab:
				'''if word exists in word2vec use vector directly'''
				dataset[sentence].vectors.append(model[word])
				w2v+=1

			elif len(wordnet_synonyms)!=0:
				'''if word does not exist in word2vec, find synonym using wordnet'''
				
				for i in wordnet_synonyms:
					if i in model.vocab:
						#use vector of similar word from wordnet
						dataset[sentence].vectors.append(model[i])
						break
				wn+=1
			
			elif word in conceptnet_embeddings_index:
				''' if word is not in word2vec, wordnet then use conceptnet'''
				c+=1
				dataset[sentence].vectors.append(conceptnet_embeddings_index[word])

			else:
				'''words un-recognised are assigned random vectors from word2vec'''
				
				#print(word)
				dataset[sentence].vectors.append(random.choice(model.wv.index2entity))

	print('\nTotal words:',k,'\nWords recognised by Word2Vec:',w2v)
	print('Words recognised by Wordnet:',wn,'\nWords recognised by ConceptNet:',c)
	print('Words not recognised:',k-(w2v+wn+c))
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
	'''removes special characters, numbers and stop words which are not of any value to the computation '''
	words=re.sub('[^A-Za-z]+', ' ', text) #removes special characters
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

#conda install -c anaconda gensim
#conda config --set auto_activate_base False
