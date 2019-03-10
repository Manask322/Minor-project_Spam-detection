# Minor-project_Spam-detection

Requirements:
1. GoogleNews-vectors-negative300.bin data file
2. SMSSpamCollection dataset

Output format:
for each tweet(SMS_data object), following attrs exist
1. label: indicates spam or ham
2. words: list of words extracted from tweet
3. vectors: list of vectors of each word in tweet
            word vector has 300 dimensions constructed using Word2Vec(gensim), WordNet,ConceptNet
