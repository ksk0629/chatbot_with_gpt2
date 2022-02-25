import pickle

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer
# 1. Tokenize all training data
# 2. Get the list of unique tokens.


class MySeq2Seq(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x


class Seq2SeqWithTokenizer(tf.keras.Model):
    """Sequence to sequence class"""

    def __init__(self, tokenizer_name: str) -> None:
        """
        Parameter
        ---------
        tokenizer_name : str
        """
        super().__init__(self)

        self.tokenizer_name = tokenizer_name
        self.__tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.__input = "input"
        self.__output = "output"

    @property
    def tokenier(self):
        return self.__tokenizer

    @property
    def input(self):
        return self.__input

    @property
    def output(self):
        return self.__output

    @propery
    def vocab_size(self):

    def encode(self):
        return self.tokenizer.encode

    def decode(self):
        return self.tokenizer.batch_decode

    def split_input_target(self, sequence: str):
        """"""
        input_sequence = sequence[:-1]
        target_sequence = sequence[1:]

        return input_sequence, target_sequence

    def create_dataset(self, dataset_pk_path: str, batch_size: int=64, buffer_size: int=10000):
        """Train this model.

        Parameter
        ---------
        dataset_pk_path : str
            path to training dataset,
            which must be a pickle file and pandas.DataFrame
            whose number of columns are two, one of them is input,
            another is output.

        Raises
        ------
        ValueError
            if the number of columns of training data, which is pandas.DataFrame, is not two
            if columns of training dataset is not [self.input, self.output]
        """
        with open(dataset_pk_path, "rb") as f:
            train_df = pickle.load(f)

        # Check whether the training data is valid
        train_columns = train_df.columns
        if len(train_columns) != 2:
            raise ValueError("Training dataset must has two columns.")
        if train_columns != [self.input, self.output]:
            raise ValueError(f"Columns of Training dataset must be {[self.input, self.output]}")

        # Get vocablary
        texts = list(np.concatenate([train_df[self.input].values, train_df[self.output].values]))
        tokenized_texts = [self.tokenize(text) for text in texts]
        self.__vocab = list(np.unique(tokenized_texts))

        # Make dataset whose data is a sequence of numerical ids
        texts = [input_text + "[SEP]" + output_text for input_text, output_text in zip(train_df[self.input].values, train_df[self.output].values)]
        ids = [self.encode(text) for text in dataset_texts]
        dataset_ids = tf.data.Dataset.from_tensor_slices(ids)
        sequences = dataset_ids(1024, drop_remainder=True)
        dataset = sequences.map(self.split_input_target)
        dataset = (dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

        self.__dataset = dataset

    def create_network(self, vocab_size, rnn_units: int=1024):
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.encode(x)
        if states is None:
          states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
          return x, states
        else:
          return x