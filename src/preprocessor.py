import argparse
from typing import List, Optional

import pandas as pd
import yaml


class LinePreProcessor():
    """Preprocessor for line history"""

    def __init__(self, input_username: str, output_username: str, target_year_list: List[str]) -> None:
        self.__data = []
        self.__cleaned_data = []
        self.__cleaned_frame = pd.DataFrame(index=[])
        self.__modified_cleaned_frame = pd.DataFrame(index=[])
        self.input_username = input_username
        self.output_username = output_username
        
        new_target_year_list = []
        for year in target_year_list:
            if year[0] == "/":
                new_target_year_list.append(str(year))
            else:
                new_target_year_list.append("/" + str(year))
        self.target_year_list = new_target_year_list

        self.__invalid_keywords = [
            "http",
            "unsent a message",
            "Missed call",
            "Canceled call",
            "Call time",
            "[Sticker]",
            "[Photo]",
            "[File]"
        ]

    @property
    def data(self) -> List[str]:
        """Raw data

        :return List[str]: raw text data
        """
        return self.__data

    @property
    def cleaned_data(self) -> List[str]:
        """Clearned data list

        :return List[str]: cleaned text data
        """
        return self.__cleaned_data

    @property
    def cleaned_frame(self) -> pd.DataFrame:
        """Cleaned data pandas.DataFrame

        :return pandas.DataFrame: cleaned text data (This has four columns, index number, time, name, and message.)
        """
        return self.__cleaned_frame

    @property
    def modified_cleaned_frame(self) -> pd.DataFrame:
        """Modified cleaned data pandas.DataFrame

        :return pd.DataFrame: modified cleaned data
        """
        return self.__modified_cleaned_frame

    @property
    def invalid_keywords(self) -> List[str]:
        """invalid keyword list

        :return List[str]: invalid keywords
        """
        return self.__invalid_keywords

    @property
    def empty(self) -> str:
        """Empty text

        :return str: empty text
        """
        return ""

    def check_to_start(self) -> None:
        """Check whether it is ready to preprocess.

        :raises AttributeError: if the length of self.data is zero
        :raises AttributeError: if self.input_username is empty, that is ""
        :raises AttributeError: if self.output_username is empty, that is ""
        :raises AttributeError: if the length of self.target_year_list is zero
        """
        # Check whether parameters valids or not
        if len(self.data) == 0:
            msg = "There is no data."
            raise AttributeError(msg)
        elif self.input_username == self.empty:
            msg = "input_username is empty."
            raise AttributeError(msg)
        elif self.output_username == self.empty:
            msg = "output_username is empty."
            raise AttributeError(msg)
        elif len(self.target_year_list) == 0:
            msg = "There is no target year."
            raise AttributeError(msg)

    def is_valid_line(self, line: str) -> bool:
        """Check whether the input line is valid.

        :param str line: text that should be checked
        :return bool: whether it is valid
        """
        for invalid_keyword in self.invalid_keywords:
            if invalid_keyword in line:
                return False
        if line == self.empty:
            return False

        return True

    def is_in_names(self, text: str) -> bool:
        """Check whether self.input_username or self.output_username is in a given text.

        :param str text: text that should be checked
        :return bool: whether there is self.input_username or self.output_username
        """
        if self.input_username in text:
            return True
        elif self.output_username in text:
            return True
        else:
            return False

    def change_notation(self, message: str) -> str:
        """Change \n to <br>.

        :param str message: original message
        :return str: message, which is processed through this function
        """

        changed_messeage = message.replace("\n", "<br>")

        return changed_messeage

    def read_text(self, text_path: str) -> None:
        """Read a text file.

        :param str text_path: path to a text file
        """
        with open(text_path, "r") as f:
            data = f.read()

        # Omit First three rows because they are meaningless
        self.__data = data.split("\n")[3:]

    def clean_data(self) -> None:
        """Clean the messages.
        """
        # Check whether or not can start cleaning the data
        self.check_to_start()

        cleaned_data = []
        for line in self.data:
            if not self.is_valid_line(line):
                continue

            # Get date and set skip flag
            if not self.is_in_names(line):
                for year in self.target_year_list:
                    if year in line:
                        index = str(line)
                        does_skip = False
                        break
                    does_skip = True  # because the lines after the next line were not sent on target year
            
            if not does_skip:
                modified_line = f"{index}\t{line}"
                cleaned_data.append(modified_line)
        
        self.__cleaned_data = cleaned_data

    def make_cleaned_frame(self) -> None:
        """Make the cleaned data as pandas.DataFrame.

        :raises AttributeError: if there are no cleaned data
        """
        # Check whether there is cleaned data or not
        if len(self.cleaned_data) == 0:
            raise AttributeError("self.clean_messages is empty.")
        
        splitted_cleaned_data = [x.split("\t") for x in self.cleaned_data]
        cleaned_frame = pd.DataFrame(splitted_cleaned_data)
        cleaned_frame.drop(index=[0], inplace=True)
    
        cleaned_frame.reset_index(inplace=True)
        cleaned_frame.drop(columns=cleaned_frame.columns[[0]], inplace=True)

        self.__cleaned_frame = cleaned_frame

    def modify_cleaned_frame(self) -> None:
        """Modify self.cleaned_frame.

        :raises AttributeError: if self.cleaned_frame is empty
        """
        # Check whether there is cleaned frame or note
        if self.cleaned_frame.empty:
            raise AttributeError("self.cleaned_frame is empty.")
        
        cleaned_frame = self.cleaned_frame

        # Set the first line as one from self.input_username
        for index, line in cleaned_frame.iterrows():
            _, _, name, _ = line
            if name != self.input_username:
                cleaned_frame.drop(cleaned_frame.index[[0]], inplace=True)
            else:
                break

        message_dict = {self.input_username: [], self.output_username: []}
        previous_name = ""
        for index, line in cleaned_frame.iterrows():
            _, _, name, message = line

            if previous_name == name:
                message_dict[name][-1] = f"{message_dict[name][-1]}<br>{self.change_notation(str(message))}"
            else:
                if name is None:  # corresponding to date line
                    continue
                message_dict[name].append(self.change_notation(str(message)))
                previous_name = name

        self.__modified_cleaned_frame = pd.DataFrame(message_dict)

    def save_as_pickle(self, output_path: str) -> None:
        """Save modified cleaned frame as pickle.
        
        :param str output_path: path to preprocessed pickle data
        :raises AttributeError: if self.modified_cleaned_frame is empty
        """
        # Check whether there is modified cleaned frame or not
        if self.modified_cleaned_frame.empty:
            raise AttributeError("self.modified_cleaned_frame is empty.")

        frame = self.modified_cleaned_frame.set_axis(["input", "output"], axis=1)
        pd.to_pickle(frame, output_path)

    def run_all(self, input_path: str, output_path: str) -> None:
        """Run all processes.

        :param str input_path: path to a text file
        :param str output_path: path to preprocessed pickle data
        """
        self.read_text(input_path)
        self.clean_data()
        self.make_cleaned_frame()
        self.modify_cleaned_frame()
        self.save_as_pickle(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Line history")

    parser.add_argument("-c", "--config_yaml_path", required=False, type=str, default="preprocessor_config.yaml")
    args = parser.parse_args()

    with open(args.config_yaml_path, "r") as yaml_f:
        config = yaml.safe_load(yaml_f)
    config_line = config["line"]
    config_initial = config_line["initial"]
    config_path = config_line["path"]

    line_preprcessor = LinePreProcessor(**config_initial)
    line_preprcessor.run_all(**config_path)
