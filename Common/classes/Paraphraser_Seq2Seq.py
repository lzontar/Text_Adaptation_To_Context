import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time


# Converts the unicode file to ascii
from tensorflow.python.keras.models import load_model


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))


class Paraphraser_Seq2Seq:
    _data_absolute_path = 'C:/Luka/School/Bachelor/Bachelor\'s thesis/Text_Adaptation/Data/paraphrases/ppdb-2.0-s-all'

    def __init__(self):
        self._num_examples = 1000
        self.units = 1024
        self.input_tensor, self.target_tensor, self.inp_lang, self.targ_lang = self.load_dataset()
        self.max_length_targ, self.max_length_inp = self.target_tensor.shape[1], self.input_tensor.shape[1]
        self.input_tensor_train, self.input_tensor_val, self.target_tensor_train, self.target_tensor_val = train_test_split(self.input_tensor,
                                                                                                        self.target_tensor,
                                                                                                        test_size=0.2)
        BUFFER_SIZE = len(self.input_tensor_train)
        BATCH_SIZE = 64
        steps_per_epoch = len(self.input_tensor_train) // BATCH_SIZE
        embedding_dim = 256
        vocab_inp_size = len(self.inp_lang.word_index) + 1
        vocab_tar_size = len(self.targ_lang.word_index) + 1

        dataset = tf.data.Dataset.from_tensor_slices((self.input_tensor_train, self.target_tensor_train)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        self.encoder = Encoder(vocab_inp_size, embedding_dim, self.units, BATCH_SIZE)
        attention_layer = BahdanauAttention(10)
        self.decoder = Decoder(vocab_tar_size, embedding_dim, self.units, BATCH_SIZE)
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_paraphrase()
        # self.evaluate("sentence")

    def evaluate(self, sentence):
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                         encoder=self.encoder,
                                         decoder=self.decoder)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))

        attention_plot = np.zeros((self.max_length_targ, self.max_length_inp))

        sentence = self.preprocess_sentence(sentence)

        inputs = [self.inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=self.max_length_inp,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = checkpoint.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.targ_lang.word_index['<start>']], 0)

        for t in range(self.max_length_targ):
            predictions, dec_hidden, attention_weights = checkpoint.decoder(dec_input,
                                                                 dec_hidden,
                                                                 enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()

            result += self.targ_lang.index_word[predicted_id] + ' '

            if self.targ_lang.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result, sentence, attention_plot

    def load_model(self):
        model = load_model('./training_checkpoints.h5')

    def train_paraphrase(self):
        input_tensor, target_tensor, inp_lang, targ_lang = self.load_dataset()
        max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                        target_tensor,
                                                                                                        test_size=0.2)
        BUFFER_SIZE = len(input_tensor_train)
        BATCH_SIZE = 64
        steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
        embedding_dim = 256
        vocab_inp_size = len(inp_lang.word_index) + 1
        vocab_tar_size = len(targ_lang.word_index) + 1

        dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        self.encoder = Encoder(vocab_inp_size, embedding_dim, self.units, BATCH_SIZE)
        attention_layer = BahdanauAttention(10)
        self.decoder = Decoder(vocab_tar_size, embedding_dim, self.units, BATCH_SIZE)
        self.optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        # with open('encoder_model.json', 'w', encoding='utf8') as f:
        #     f.write(self.encoder.to_json())
        # self.encoder.save_weights('encoder_model_weights.h5')
        #
        # with open('decoder_model.json', 'w', encoding='utf8') as f:
        #     f.write(self.decoder.to_json())
        # self.decoder.save_weights('decoder_model_weights.h5')
        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)

            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask

            return tf.reduce_mean(loss_)

        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                         encoder=self.encoder,
                                         decoder=self.decoder)

        @tf.function
        def train_step(inp, targ, enc_hidden):
            loss = 0

            with tf.GradientTape() as tape:
                enc_output, enc_hidden = self.encoder(inp, enc_hidden)

                dec_hidden = enc_hidden

                dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

                    loss += loss_function(targ[:, t], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))

            variables = self.encoder.trainable_variables + self.decoder.trainable_variables

            gradients = tape.gradient(loss, variables)

            self.optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss

        EPOCHS = 2

        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        checkpoint.save('C:/Luka/School/Bachelor/Bachelor\'s thesis/Text_Adaptation/training_checkpoints/seq2seq.h5')

    def preprocess_sentence(self, w):
        w = unicode_to_ascii(w.lower().strip())

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

        w = w.strip()

        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '<start> ' + w + ' <end>'
        return w

    def read_data(self):
        lines = io.open(self._data_absolute_path, encoding='UTF-8').read().strip().split('\n')
        split_words = list(map(lambda x: x.split('|||'), lines[:self._num_examples]))
        words = list(map(lambda x: self.preprocess_sentence(x[1]), split_words[:self._num_examples]))
        paraphrases = list(map(lambda x: self.preprocess_sentence(x[2]), split_words[:self._num_examples]))

        return words, paraphrases

    def tokenize(self, lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='')
        lang_tokenizer.fit_on_texts(lang)

        tensor = lang_tokenizer.texts_to_sequences(lang)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post')

        return tensor, lang_tokenizer

    def load_dataset(self):
        # creating cleaned input, output pairs
        targ_lang, inp_lang = self.read_data()

        input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
        target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)

        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')


  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, self.units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights