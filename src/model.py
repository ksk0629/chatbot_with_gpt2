import argparse
from typing import Optional
import yaml

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ConversationalModel():
    """Conversational model class"""

    def __init__(model_path: str, tokenizer_name: str, model_name: str="model") -> None:
        """
        Parameters
        ----------
        model_path : str
        tokenizer_name : str
        model_name : str
            model name that is used in self.take_conversations() function

        """
        self.model_path = model_path
        self.model_name = model_name
        self.__tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.__model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model.to(self.device)

    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def model(self):
        return self.__model

    def reply(self, input_message: str, num_responses: int) -> str:
        """Reply to the input message.

        Parameter
        ---------
        input_message : str
        num_responses : int
        """
        actual_input_message = "<s>" + str(input_message) + "[SEP]"
        input_vector = self.tokenizer.encode(actual_input_message, return_tensors='pt').to(self.device)

        output = self.model.generate(input_vector, do_sample=True, max_length=128, num_return_sequences=num_responses,
                                     top_p=0.95, top_k=50, bad_words_ids=[[1], [5]], no_repeat_ngram_size=3)

        actual_response = ""
        for response in self.tokenizer.batch_decode(output):
            actual_response += response.split('[SEP]</s>')[1].replace('</s>', '').replace('<br>', '\n') + "\n"

        return actual_response

    def take_conversations(self, num: Optional[int]=None):
        """Talk to this model.

        Parameter
        ---------
        num : Optional[int], default None
            the number of conversations
        """
        count = 0 if num is not None else num + 1

        while True:
            print("You > ", end="")
            input_message = input()
            print()
            output_message = self.reply(input_message)
            print(f"{self.model_name} > {output_message}")

            if count == num:
                break
            else:
                count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Talk with the model")

    # Add arguments: [https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0]
    parser.add_argument("-c", "--config_yaml_path", required=False, type=str, default="mdoel_config.yaml")

    args = parser.parse_args()

    with open(args.config_yaml_path, "r") as yaml_f:
        config = yaml.safe_load(yaml_f)
    config_general = config["general"]
    config_train = config["train"]

    model = ConversationalModel(model_path=config_train["output_dir"], tokenizer_name=config_general["baseline"])

    model.take_conversations()