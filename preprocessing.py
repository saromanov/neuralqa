from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences, skipgrams

def nltk_tokenizer(sent):
  return word_tokenize(sent)