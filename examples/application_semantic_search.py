"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from sentence_transformers import SentenceTransformer
import scipy.spatial
from numpy import array
import csv


model  = "roberta-base-nli-stsb-mean-tokens"
#model = "roberta-large-nli-stsb-mean-tokens"



# NOT GOOD
#model = "bert-base-nli-stsb-mean-tokens"
#model = "roberta-large-nli-mean-tokens"
#model = "distilbert-base-nli-stsb-mean-tokens"
#model = "bert-large-nli-stsb-mean-tokens"
#model = "bert-large-nli-mean-tokens"
#model = "bert-large-nli-max-tokens"
#model = "bert-large-nli-cls-token"
#model = "distilbert-base-nli-mean-tokens"
#model = "bert-base-wikipedia-sections-mean-tokens"


# REQUIRES GPU
#model = "distiluse-base-multilingual-cased"


embedder = SentenceTransformer(model)



corpus = []
issues = []
issues_clauses = []
with open('issues_with_content_updated.csv', newline='') as csvfile:
    textreader = csv.reader(csvfile, dialect='excel')
    for row in textreader:
        corpus.append(row[2])
        issues_clauses.append(row[0])
        issues.append(row[1])

# Corpus with example sentences
"""
corpus = [
'Hello is a Pop song by Adele released in 2015.',
'Hello is a Pop song by Lionel Richie released in 1983.',
'25 is a Pop album by Adele.',
'Bad is a Pop song by Michael Jackson released in 1987.',
'Michael Bublé is a Canadian is singer.',
'Adele is a British songwriter and singer.',
'Lionel Richie is an American songwriter and singer.',
'Ozzy Osbourne is an American songwriter and singer.',
'Michael Jackson was an American songwriter and singer.',
'Beyonce is an American songwriter and singer.',
'Diana Krall is a Canadian songwriter and singer.',
'Born in the U.S.A. is a Rock song by Bruce Springsteen released in 1984',
'Rocketman is a Soft Rock song by Elton John released in 1972.',
'Elton John is a British songwriter and singer.',
'Billy Joel is an American songwriter and singer.',
'Billie Jean is a Pop song by Michael Jackson released in 1982.', 
'Bad 25 is a Pop Album by Michael Jackson.',
'Bruce Springsteen is an American songwriter and singer.',
'With or Without You is an Alternative Rock Album by U2.',
'Hello Hello! is a Kids Songs, released in the year 2017.',
'U2 is an Irish rock band.']
"""

corpus_embeddings = embedder.encode(corpus)

clauses = []
before_text =[]
after_text = []
added_text = []
with open('output.csv', newline='') as csvfile:
    textreader = csv.reader(csvfile, dialect='excel')
    for row in textreader:
    	if row[3]!= row[2]:
    		clauses.append(row[0])
    		before_text.append(row[2])
    		after_text.append(row[3])
    		added_text.append(row[4])


# Query sentences:
#queries = ['born','Born in the U.S.A.','Grammy Awards winners.','The Boss.', 'Michael.','Hello.', 'Hello by Adele.', 'Adele.' ,'Hello by Lionel.', '25.', 'Pop music.','R&B Soul','Rock music.','From the 1980\'s.', 'Artists.', 'Piano','Rocket Man.', 'Billy Jeans','children\'s songs', 'The King of Pop']
after_text_embeddings = array(embedder.encode(after_text))
added_text_embeddings = array(embedder.encode(added_text))
issues_embeddings = array(embedder.encode(issues))

query_embeddings=(after_text_embeddings+added_text_embeddings)/2

#query_embeddings = after_text_embeddings


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 2
for before, after, added_embedding, query_embedding, clause in zip(before_text, after_text, added_text_embeddings, query_embeddings, clauses):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Clause:", clause)
    print("Before:", before)
    print("After:", after)
    print("\nTop ", closest_n," most similar sentences in corpus:")
 
    max_score=0
    max_issue=[]
    for idx, distance in results[0:closest_n]:
    	if ((1-distance) > 0.5):
    		print("Clause: ", issues_clauses[idx])
    		print("Issue: ",issues[idx].strip())
    		print(corpus[idx].strip(), "(After Text Score: %.4f)" % (1-distance))
    		'''
    		inner_distance = scipy.spatial.distance.cdist([added_embedding],[issues_embeddings[idx]],"cosine")[0]
    		print("Issue: ",issues[idx].strip(), "(Issue Score: %.4f)" % (1-inner_distance))
    		current_score=(0.35*(1-inner_distance)+0.65*(1-distance))
    		print("Combined Score: %.4f" % current_score )
    		if current_score > max_score:
    			max_score = current_score
    			max_issue = issues[idx].strip()

    print("Winner: ", max_issue, "Score: %.4f" % max_score)

'''
    input('Press enter to continue: ')



