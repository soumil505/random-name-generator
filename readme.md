<h1>Random name generator</h1>
A markov-chain based model that can be trained to generate random fake names, ideal for creating fake fantasy names.
<br>
<br>
Even though the model is trained on names, it can be trained to generate any text by replacing the text in first_names.all.txt<br>
<br>
<h2>Usage</h2><br>
	from word_generator import WordGenerator<br>
	model=WordGenerator()<br>
	model.train(word_list,n_gram=3,end_ngram_length=2)<br>
	print("generated word of (approximate) length",5,"is",model.generate(5))<br>
<br><br>
Name data was taken from https://github.com/philipperemy/name-dataset 