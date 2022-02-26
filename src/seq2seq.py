import pickle
import random

import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
import transformer


def train_seq2seq(pk_path: str, tokenizer_name: str, batch_size: int, epochs: int, dimensionality: int,
                  output_path: str):
    with open(pk_path, "rb") as pk_f:
        raw_dataset = pickle.load(pk_f)

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

    input_texts = self.raw_dataset[self.input].values
    output_texts = self.raw_dataset[self.output].values

    paris = list(zip(input_texts, output_texts))
    random.shuffle(pairs)

    input_docs = []
    target_docs = []
    input_tokens = set()
    target_tokens = set()
    for line in pairs:
        input_doc, target_doc = line[0], line[1]
        # Appending each input sentence to input_docs
        input_docs.append(input_doc)
        # Splitting words from punctuation  
        target_doc = " ".join(tokenizer.tokenize(target_doc))
        # Redefine target_doc below and append it to target_docs
        target_doc = '<START> ' + target_doc + ' <END>'
        target_docs.append(target_doc)
      
        # Now we split up each sentence into words and add each unique word to our vocabulary set
        for token in tokenizer.tokenize(input_doc):
            if token not in input_tokens:
                input_tokens.add(token)
        for token in target_doc.split():
            if token not in target_tokens:
                target_tokens.add(token)
    input_tokens = sorted(list(input_tokens))
    target_tokens = sorted(list(target_tokens))
    num_encoder_tokens = len(input_tokens)
    num_decoder_tokens = len(target_tokens)
    
    input_features_dict = dict([(token, i) for i, token in enumerate(input_tokens)])
    target_features_dict = dict([(token, i) for i, token in enumerate(target_tokens)])

    reverse_input_features_dict = dict((i, token) for token, i in input_features_dict.items())
    reverse_target_features_dict = dict((i, token) for token, i in target_features_dict.items())

    tokenizer = transformer.AutoTokenizer.from_pretrained(tokenizer_name)

    max_encoder_seq_length = max([len(tokenizer.tokenize(input_doc)) for input_doc in input_docs])
    max_decoder_seq_length = max([len(tokenizer.tokenize(target_doc)) for target_doc in target_docs])

    encoder_input_data = np.zeros((len(input_docs), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    decoder_input_data = np.zeros((len(input_docs), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_docs), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
        for timestep, token in enumerate(tokenizer.tokenize(input_doc)):
            #Assign 1. for the current line, timestep, & word in encoder_input_data
            encoder_input_data[line, timestep, input_features_dict[token]] = 1.
        
        for timestep, token in enumerate(tokenizer.tokenize(target_doc)):
            decoder_input_data[line, timestep, target_features_dict[token]] = 1.
            if timestep > 0:
                decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.

    #Encoder
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder_lstm = LSTM(dimensionality, return_state=True)
    encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
    encoder_states = [state_hidden, state_cell]
    #Decoder
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
    decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    #Model
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    #Compiling
    training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')
    #Training
    training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size = batch_size, epochs = epochs, validation_split = 0.2)
    training_model.save(output_path)


def talk_with_model(model_path: str, latent_dim: int):
    training_model = load_model(model_path)
    encoder_inputs = training_model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_hidden = Input(shape=(latent_dim,))
    decoder_state_input_cell = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

    decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_hidden, state_cell]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    def decode_response(test_input):
        #Getting the output states to pass into the decoder
        states_value = encoder_model.predict(test_input)
        #Generating empty target sequence of length 1
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        #Setting the first token of target sequence with the start token
        target_seq[0, 0, target_features_dict['<START>']] = 1.
        
        #A variable to store our response word by word
        decoded_sentence = ''
        
        stop_condition = False
        while not stop_condition:
            #Predicting output tokens with probabilities and states
            output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)
            #Choosing the one with highest probability
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_features_dict[sampled_token_index]
            decoded_sentence += " " + sampled_token
            #Stop if hit max length or found the stop token
            if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
              stop_condition = True
            #Update the target sequence
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
            #Update states
            states_value = [hidden_state, cell_state]
        return decoded_sentence

    class ChatBot:
        negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
        exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")
        
        #Method to start the conversation
        def start_chat(self):
            user_response = input("Hi, I'm a chatbot trained on random dialogs. Would you like to chat with me?\n")

            if user_response in self.negative_responses:
                print("Ok, have a great day!")
                return
            self.chat(user_response)
        
        #Method to handle the conversation
        def chat(self, reply):
            while not self.make_exit(reply):
                reply = input(self.generate_response(reply)+"\n")
        
            #Method to convert user input into a matrix
            def string_to_matrix(self, user_input):
                tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
                user_input_matrix = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
                for timestep, token in enumerate(tokens):
                    if token in input_features_dict:
                        user_input_matrix[0, timestep, input_features_dict[token]] = 1.
                return user_input_matrix
      
        #Method that will create a response using seq2seq model we built
        def generate_response(self, user_input):
            input_matrix = self.string_to_matrix(user_input)
            chatbot_response = decode_response(input_matrix)
            #Remove <START> and <END> tokens from chatbot_response
            chatbot_response = chatbot_response.replace("<START>",'')
            chatbot_response = chatbot_response.replace("<END>",'')
            return chatbot_response

        #Method to check for exit commands
        def make_exit(self, reply):
            for exit_command in self.exit_commands:
                if exit_command in reply:
                    print("Ok, have a great day!")
                    return True
        return False

        chatbot = ChatBot()
        chatbot.start_chat()
