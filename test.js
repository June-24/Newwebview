import gensim
from gensim.models import Word2Vec

# Sample sentences
sentences = [
    ['this', 'is', 'a', 'sample', 'sentence'],
    ['gensim', 'is', 'a', 'great', 'library', 'for', 'topic', 'modeling'],
    ['we', 'are', 'testing', 'if', 'gensim', 'is', 'installed', 'properly'],
    ['word2vec', 'is', 'a', 'popular', 'algorithm', 'for', 'word', 'embeddings']
]

# Train a Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Save the model
model.save("test_word2vec.model")

# Load the model
model = Word2Vec.load("test_word2vec.model")

# Check the model by finding the most similar words
similar_words = model.wv.most_similar('gensim', topn=5)
print("Most similar words to 'gensim':")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")

