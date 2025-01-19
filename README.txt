----- READ ME ------ 

Video Game Information Retrieval System:

A Python based IR system designed to efficiently retrieve the relevant documents from a collection of video game HTML files.

This project is split into two different files, here is what each do:

index.py - Used for building the Inverted Index Model by processing the raw text of HTML documents.

search.py - Handles the retrieval process including preprocessing, vectorisation, and ranking

Features:
- Single and multi term queries are supported
- Uses TF-IDF weighting and the vector space model
- Includes preprocessing techniques like stemming and lemmatization
- Provides ranked search results and precision@10 evaluations

The dataset used is videogame-labels.csv. 

Prerequisites: 

Python 3

Libraries to install: (use pip install x)
NLTK
FuzzyWuzzy


How to run: 

1 - Run index.py  in the terminal 
python index.py

2- Run search.py in the terminal
python search.py

3 - Enter a search query of your choice and press ENTER

4 - View the ranked search results in the console

5- To exit, type 'quit'
