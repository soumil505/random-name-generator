from word_generator import WordGenerator
import re
import pickle

model=WordGenerator()
word_list=open("first_names.all.txt","r").read()
word_list=re.sub(r"[^a-zA-Z0-9@#$%\^\\/&\*\(\):;\?!'\-\n]","",word_list).split("\n")

model.train(word_list,3,2)
pickle.dump( model, open( "model.pkl", "wb" ) )
for approx_length in range(3,15):
	print("generated word of (approximate) length",approx_length,"is",model.generate(approx_length))
