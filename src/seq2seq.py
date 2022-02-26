import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd
from tensorflow import keras
import transformer


class Seq2seq():

    def __init__(self, tokenizer_name: str):
        self.__raw_dataset = None
        self.__dataset = None
        self.__model = None
        self.tokenizer_name = tokenizer_name
        self.__tokenizer = transformer.AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.__special_tokens = list(self.tokenizer.special_tokens_map.values())

        self.input = "input"
        self.output = "output"

    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def special_tokens(self):
        return self.__special_tokens

    @property
    def vocab_size(self):
        return self.__tokenizer.vocab_size

    @property
    def raw_dataset(self):
        return self.__raw_dataset

    @property
    def dataset(self):
        return self.__dataset

    @property
    def dataset_size(self):
        return len(self.dataset[self.input])

    @property
    def model(size):
        return self.__model

    def tokenize(self, text: str):
        return tokenizer.tokenize(text)

    def encode(self, text: str):
        return tokenizer.encode(text)

    def decode(self, text: str):
        return tokenizer.batch_decode(text)

    def get_dataset_from_pk(self, dataset_path: str):
        if pathlib.Path(dataset_path).suffix != "pk":
            error_msg = "Dataset path must be pickle file."
            raise ValueError(error_msg)

        with open(pk_file, "rb") as pk_f:
            raw_dataset = pickle.load(pk_f)

        if type(dataset) != pd.DataetFrame:
            error_msg = f"Dataset must be pandas.DatasetFrame,\nbut it's {type(dataset)}."
            raise ValueError(error_msg)

        columns = raw_dataset.columns
        if not (self.input in columns and self.output in columns):
            error_msg = "Dataset must have 'input' and 'output columns"
            raise ValueError(error_msg)

        self.__raw_dataset = raw_dataset

    def create_dataset_from_pk(self, dataset_path: str, shuffle=True):
        self.get_dataset_from_pk(dataset_path)

        input_texts = self.raw_dataset[self.input].values
        output_texts = self.raw_dataset[self.output].values

        preprocessed_input_tokenized = []
        preprocessed_output_tokenized = []
        for input_text, output_text in zip(input_texts, output_texts):
            if self.decode(self.encode(input_text)) in self.special_tokens:
                continue
            elif self.decode(self.encode(output_text)) in self.special_tokens:
                continue
            else:
                preprocessed_input_tokenized.append(self.tokenize(input_text))
                preprocessed_output_tokenized.append(self.tokenize(output_text))

        error_msg = "Preprocessed input and output data must be same size."
        assert len(preprocessed_input_tokenized) == len(preprocessed_output_tokenized), error_msg

        dataset = {
            self.input: preprocessed_input_tokenized,
            self.output: preprocessed_output_tokenized
            }

        self.__dataset = dataset

    def setup_training(self, max_seq_length: int, batch_size: int, epochs: int):
        max_input_length = max([len(text) for text in self.dataset[self.input]])
        max_output_length = max([len(text) for text in self.dataset[self.output]])
        max_length_from_dataset = max[max_input_length, max_output_length]
        if max_seq_length < max_length_from_dataset:
            warning_msg = f"max_seq_length is set as {max_length_from_dataset}"
            warnings.warn(warning_msg)
            max_seq_length = max_length_from_dataset

        self.encoder_input_data = np.zeros(self.dataset_size, max_seq_length, self.vocab_size, dtype="float32")
        self.decoder_input_data = np.zeros(self.dataset_size, max_seq_length, self.vocab_size, dtype="float32")
        self.decoder_target_data = np.zeros(self.dataset_size, max_seq_length, self.vocab_size, dtype="float32")

        for line, (input_text, target_text) in enumerate(zip(self.dataset[self.input], self.dataset[self.output])):
            for timestep, token in enumerate(input_text):
                self.encode_input_data[line, timestep, self.encode(token)] = 1.

            for timestep, token in enumerate(target_text):
                self.decoder_input_Data[line, timestep, self.encode(token)] = 1.
                if timestep > 0:
                    self.decoder_target_data[line, timestep-1, self.encode(token)] = 1.

        self.batch_size = batch_size
        self.epochs = epochs

    def create_network(self, lstm_dimension: int):
        self.encoder_inputs = keras.layers.Input(shape=(None, self.vocab_size))
        self.encoder_lstm = keras.layers.LSTM(lstm_dimension, return_state=True)
        _, state_hidden, state_cell = self.encoder_lstm(self.encoder_inputs)
        encoder_states = [state_hidden, state_cell]

        self.decoder_inputs = keras.layers.Input(shape=(None, self.vocab_size))
        self.decoder_lstm = keras.layers.LSTM(lstm_dimension, return_sequences=True, return_state=True)
        decoder_outputs, decoder_state_hidden, decoder_state_cell = self.decoder_lstm(self.decoder_inputs, initial_state=encoder_states)
        self.decoder_dense = keras.layers.Dense(self.vocab_size, activation='softmax')
        decoder_outputs = self.decoder_dense(decoder_outputs)

        self.__model = keras.models.Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)

    def train_model(self, save_path: str=None):
        if self.model is None:
            raise ValueError("There is no model.")
        self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"], sample_weight="temporal")
        self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.2)

        if save_path is not None:
            self.model.save(save_path)
