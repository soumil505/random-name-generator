import sys
from random import choices,random
import numpy as np

class WordGenerator:
	def __init__(self):
		self.word_list=[]
		self.N=2
		self.markov_dict={}
		self.start_ngrams=[]
		self.start_ngrams_probs=[]
		self.near_start={}
		self.near_end={}
		self.end_ngram_length=2
		self.end_ngrams=[]
		self.end_ngrams_probs=[]
	
	def create_markov_dict(self,word_list,n_gram=2):
		words="~".join(word_list).lower()
		# words=re.sub(r"[^a-zA-Z0-9@#$%\^\\/&\*\(\):;\?!'\-]","",words.lower())
		fin=len(words)-n_gram-2
		markov_dict={}
		for index in range(0,fin):
			prog = (index)/(fin-1)*100
			if int(prog)%2==0:
				sys.stdout.write("\r"+" "*50+"\r"+str(round(prog,2))+"% done")
			ngram=words[index:index+n_gram]
			if "~" in ngram:
				continue
			next_char=words[index+n_gram]
			if next_char=="~":
				continue
			if ngram not in markov_dict.keys():
				markov_dict[ngram]={"count":words.count(ngram,index)}
			
			if next_char not in markov_dict[ngram].keys():
				markov_dict[ngram][next_char]=1
			else:
				markov_dict[ngram][next_char]+=1
		print()
		return markov_dict
	
	def get_start_ngrams(self,word_list,n_gram=2):
		ngrams=[word[:n_gram].lower() for word in word_list if len(word)>=n_gram]
		ngrams=[word for word in ngrams if len(word) >=n_gram]
		unique=list(set(ngrams))
		denominator=len(ngrams)
		return unique,[ngrams.count(word)/denominator for word in unique]
		
	def get_end_ngrams(self,word_list,n_gram=2):
		ngrams=[word[-n_gram:].lower() for word in word_list if len(word)>=n_gram]
		ngrams=[word for word in ngrams if len(word) >=n_gram]
		all_ngrams=''.join(word_list).lower()
		unique=list(set(ngrams))
		return unique,[ngrams.count(w)/all_ngrams.count(w) for w in unique]
	
	def get_near_start_prob_dict(self,word_list):
		prob_dict={}
		start_chars=''.join([word[:len(word)//2] for word in word_list]).lower()
		unique_chars=list(set(start_chars))
		denominator=len(start_chars)
		return {char:start_chars.count(char)/denominator for char in unique_chars}
		
	def get_near_end_prob_dict(self,word_list):
		prob_dict={}
		start_chars=''.join([word[len(word)//2:] for word in word_list]).lower()
		unique_chars=list(set(start_chars))
		denominator=len(start_chars)
		return {char:start_chars.count(char)/denominator for char in unique_chars}
		
	
	
	def train(self,word_list,n_gram,end_ngram_length):
		self.word_list=word_list
		self.N=n_gram
		self.end_ngram_length=end_ngram_length
		print("\033[34mcalculating markov chain probabilities \033[0m")
		self.markov_dict=self.create_markov_dict(self.word_list,self.N)
		print("\033[34mfinishing up \033[0m")
		self.start_ngrams,self.start_ngrams_probs=self.get_start_ngrams(self.word_list,self.N)
		self.near_start=self.get_near_start_prob_dict(self.word_list)
		self.near_end=self.get_near_end_prob_dict(self.word_list)
		self.end_ngrams,self.end_ngrams_probs=self.get_end_ngrams(self.word_list,self.end_ngram_length)
		print("\033[34mdone training \033[0m")
		
	def generate(self,approx_length,limit=15):
		def isend(ngram,end_ngrams,end_ngrams_probs,probab_end=0.5):
			if ngram not in end_ngrams:
				return 0
			else:
				return end_ngrams_probs[end_ngrams.index(ngram)]>probab_end #the smaller this number, the shorter the word
		start=choices(self.start_ngrams,weights=self.start_ngrams_probs)[0]
		generated=start
		approx_length-=len(start)
		if self.end_ngram_length<0:
			self.end_ngram_length=len(start)
		for idx in range(limit):
			start=generated[-len(start):]
			prob_dict=self.markov_dict[start].copy()
			count=prob_dict["count"]
			prob_dict.pop("count")
			options=list(prob_dict.keys())
			probs=[prob_dict[choice]/count for choice in options]
			#update probs
			if idx<approx_length//2:
				weights=[self.near_start[char] for char in options]
				probs=np.multiply(probs,weights)
				probs=list(probs/np.sum(probs))
			if idx>=approx_length//2:
				weights=[self.near_end[char] for char in options]
				probs=np.multiply(probs,weights)
				probs=list(probs/np.sum(probs))
			generated+=choices(options,weights=probs)[0]
			if isend(generated[-self.end_ngram_length:],self.end_ngrams,self.end_ngrams_probs,probab_end=1-1.1*idx/(approx_length+1e-10)):
				return generated
		return generated