from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Merge, Permute, Dropout, Layer
from keras.layers.core import Highway
from keras.layers import LSTM, GRU
from keras.layers.advanced_activations import PReLU
from keras.utils.data_utils import get_file
from functools import reduce
import tarfile
import numpy as np
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

class Model:
	def __init__(self, trainX, textY):
		self.trainX = trainX
		self.textX = textY
		self.answerModel = None
		self._input_encoder = None
		self._question_encoder = None

	def input_encoder(self, dropout=0.3):
		# embed the input sequence into a sequence of vectors
		input_encoder_m = Sequential()
		input_encoder_m.add(Embedding(input_dim=self.vocab_size,
                              output_dim=64,
                              input_length=self.story_maxlen))
		input_encoder_m.add(Dropout(dropout))
		self._input_encoder = input_encoder

	def question_encoder(self, dropout=0.3):
		question_encoder = Sequential()
		question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=query_maxlen))
		question_encoder.add(Dropout(dropout))
		self._question_encoder = question_encoder


	def train(self, method='rmslprop'):
		if self.answerModel is None:
			raise Exception("Answer model was not initialized")

		self.answerModel.compile(optimizer=method, loss='categorical_crossentropy',
               metrics=['accuracy'])
		answer.fit([inputs_train, queries_train, inputs_train], answers_train,
           	batch_size=32,
           	nb_epoch=120,
           	validation_data=([inputs_test, queries_test, inputs_test], answers_test))

	def construct():

		match = Sequential()
		match.add(Merge([input_encoder_m, question_encoder],
                	mode='dot',
                	dot_axes=[2, 2]))
		# output: (samples, story_maxlen, query_maxlen)
		# embed the input into a single vector with size = story_maxlen:
		input_encoder_c = Sequential()
		input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=query_maxlen,
                              input_length=story_maxlen))
		input_encoder_c.add(Dropout(0.3))
		# output: (samples, story_maxlen, query_maxlen)
		# sum the match vector with the input vector:
		response = Sequential()
		response.add(Merge([match, input_encoder_c], mode='sum'))
		# output: (samples, story_maxlen, query_maxlen)
		response.add(Permute((2, 1)))  # output: (samples, query_maxlen, story_maxlen)

		# concatenate the match vector with the question vector,
		# and do logistic regression on top
		answer = Sequential()
		answer.add(Merge([response, question_encoder], mode='concat', concat_axis=-1))
		# the original paper uses a matrix multiplication for this reduction step.
		# we choose to use a RNN instead.
		answer.add(GRU(32))
		# one regularization layer -- more would probably be needed.
		answer.add(Dropout(0.4))
		answer.add(Dense(vocab_size))
		answer.add(PReLU())
		# we output a probability distribution over the vocabulary
		answer.add(Activation('softmax'))

		answer.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])
		# Note: you could use a Graph model to avoid repeat the input twice
		answer.fit([inputs_train, queries_train, inputs_train], answers_train,
           	batch_size=32,
           	nb_epoch=120,
           	validation_data=([inputs_test, queries_test, inputs_test], answers_test))