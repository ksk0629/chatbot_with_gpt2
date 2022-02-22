import argparse
import re
from typing import Optional
import yaml

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ConversationalModel():
    """Conversational model class"""

    def __init__(self, model_path: str, tokenizer_name: str, model_name: str="model") -> None:
        """
        Parameters
        ----------
        model_path : str
            path to a model file
        tokenizer_name : str
            name of a tokenizer, which is used in reply function
        model_name : str
            model name that is used in self.take_conversations() function
        """
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.__tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.__model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model.to(self.device)
        self.model_name = model_name

    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def model(self):
        return self.__model

    def delete_unnecessary_words(self, message: str) -> str:
        """Delete unnecessary words.

        Parameter
        ---------
        message : str
            original message, which is outputted from this model

        Return
        ------
        message_deleted_unnnecessary_words : str
            message, which is modified through this function
        """
        message_deleted_unnnecessary_words = message.split('[SEP]</s>')[1]

        message_deleted_unnnecessary_words = re.sub(r"\n", "", message_deleted_unnnecessary_words)

        message_deleted_unnnecessary_words = message_deleted_unnnecessary_words.replace('</s>', '')
        message_deleted_unnnecessary_words = message_deleted_unnnecessary_words.replace('<br>', '\n')
        message_deleted_unnnecessary_words = message_deleted_unnnecessary_words.replace('<br/>', '\n')
        message_deleted_unnnecessary_words = message_deleted_unnnecessary_words.replace('<br:', '\n')
        message_deleted_unnnecessary_words = message_deleted_unnnecessary_words.replace('<br)', '\n')
        message_deleted_unnnecessary_words = message_deleted_unnnecessary_words.replace('<br', '\n')
        message_deleted_unnnecessary_words = message_deleted_unnnecessary_words.replace('<br;', '\n')
        message_deleted_unnnecessary_words = message_deleted_unnnecessary_words.replace('</br;', '\n')
        message_deleted_unnnecessary_words = message_deleted_unnnecessary_words.replace('<a', '\n')
        
        message_deleted_unnnecessary_words = message_deleted_unnnecessary_words.replace('[<unk>hoto', '')
        message_deleted_unnnecessary_words = message_deleted_unnnecessary_words.replace('<<unk>hoto', '')
        message_deleted_unnnecessary_words = message_deleted_unnnecessary_words.replace('/', '')

        message_deleted_unnnecessary_words = re.sub(r"\[*\]", "", message_deleted_unnnecessary_words)
        
        message_deleted_unnnecessary_words = message_deleted_unnnecessary_words + "\n"

        return message_deleted_unnnecessary_words

    def reply(self, input_message: str, max_length: int=128) -> str:
        """Reply to the input message.

        Parameter
        ---------
        input_message : str
            message, which is inpuuted to this model
        max_length : int
            maximum length of the output message
            Note that, the length is about token not letter or word.

        Return
        ------
        actual_response : str
            message, which is outputted from this model
        """
        actual_input_message = "<s>" + str(input_message) + "[SEP]"
        input_vector = self.tokenizer.encode(actual_input_message, return_tensors='pt').to(self.device)

        output = self.model.generate(input_vector, do_sample=True, max_length=128, num_return_sequences=1,
                                     pad_token_id=2, top_p=0.95, top_k=50, bad_words_ids=[[1], [5]], no_repeat_ngram_size=3)

        actual_response = ""
        for response in self.tokenizer.batch_decode(output):
            actual_response += self.delete_unnecessary_words(response)

        return actual_response

    def take_conversations(self, num: Optional[int]=None) -> None:
        """Talk with this model.

        Parameter
        ---------
        num : Optional[int], default None
            the number of conversations
        """
        count = 0

        while True:
            print("You >")
            input_message = input()
            output_message = self.reply(input_message)
            print(f"{self.model_name} >")
            print(f"{output_message}")

            if count == num:
                break
            else:
                count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Talk with a model")

    # Add arguments: [https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0]
    parser.add_argument("-c", "--config_yaml_path", required=False, type=str, default="model_config.yaml")

    args = parser.parse_args()

    with open(args.config_yaml_path, "r") as yaml_f:
        config = yaml.safe_load(yaml_f)
    config_general = config["general"]
    config_train = config["train"]

    model = ConversationalModel(model_path=config_train["output_dir"], tokenizer_name=config_general["basemodel"])

    model.take_conversations()
