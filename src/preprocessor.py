import argparse
from typing import List, Optional
import yaml

import pandas as pd


class LinePreProcessor():
    """Preprocessor for line history"""

    def __init__(self, input_username: str, output_username: str, target_year_list: List[str]):
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
        ]

        self.__invalid_output_keywords = [
            "[Sticker]",
            "[Photo]",
            "[File]"
        ]

    @property
    def data(self) -> List[str]:
        return self.__data

    @property
    def cleaned_data(self) -> List[str]:
        return self.__cleaned_data

    @property
    def cleaned_frame(self) -> pd.DataFrame:
        # This has four columns, index number, time, name, and message
        return self.__cleaned_frame

    @property
    def modified_cleaned_frame(self) -> pd.DataFrame:
        return self.__modified_cleaned_frame

    @property
    def invalid_keywords(self) -> List[str]:
        return self.__invalid_keywords

    @property
    def invalid_output_keywords(self) -> List[str]:
        return self.__invalid_output_keywords

    @property
    def empty(self) -> str:
        return ""

    def check_to_start(self) -> None:
        """Check whether it is ready to preprocess.

        Raises
        ------
        AttributeError :
            if the length of self.data is zero
            if self.input_username is empty, that is ""
            if self.output_username is empty, that is ""
            if the length of self.target_year_list is zero
        """
        if len(self.data) == 0:
            raise AttributeError("There is no data.")
        elif self.input_username == self.empty:
            raise AttributeError("input_username is empty.")
        elif self.output_username == self.empty:
            raise AttributeError("output_username is empty.")
        elif len(self.target_year_list) == 0:
            raise AttributeError("There is no target year.")

    def is_valid_line(self, line: str) -> bool:
        """Check whther the input line is valid.

        Parameter
        ---------
        line : str

        Return
        ------
        bool
            whether it is valid
        """
        for invalid_keyword in self.invalid_keywords:
            if invalid_keyword in line:
                return False
        if line == self.empty:
            return False

        return True

    def is_in_names(self, text: str) -> bool:
        """Check whther self.input_username or self.output_username is in the input text.

        Parameter
        ---------
        text : str

        Return
        ------
        bool
            whether there is the names
        """
        if self.input_username in text:
            return True
        elif self.output_username in text:
            return True
        else:
            return False

    def is_invalid_output(self, line: str) -> bool:
        """Check if the line is from the output user and invalid.

        Parameter
        ---------
        line : str

        Return
        ------
        bool
            whether it id from the output user and invalid
        """
        splitted_line = line.split("\t")

        if len(splitted_line) == 1:  # only message
            return False
        
        name = splitted_line[0]
        message = splitted_line[1]

        if name == self.output_username:
            for invalid_output_keyword in self.invalid_output_keywords:
                if invalid_output_keyword in message:
                    return True


        return False

    def change_notation(self, message: str) -> str:
        """Change some notations.

        Parameter
        ---------
        message : str

        Return
        ------
        changed_message : str
        """
        changed_messeage = message.replace("\n", "<br>")
        return changed_messeage

    def read_text(self, text_path: str) -> None:
        """Read a text file.

        Parameter
        ---------
        text_path : str
        """
        with open(text_path) as f:
            data = f.read()

        self.__data = data.split("\n")[3:]

    def clean_data(self) -> None:
        """Clean the messages."""
        # Raise AttributeError if it is not ready
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
            
            if not (does_skip or self.is_invalid_output(line)):
                modified_line = f"{index}\t{line}"
                cleaned_data.append(modified_line)
        
        self.__cleaned_data = cleaned_data

    def make_cleaned_frame(self) -> None:
        """Make the cleaned data as pandas.DataFrame.

        Raise
        -----
        AttributeError :
            if there are no cleaned data
        """
        if len(self.cleaned_data) == 0:
            raise AttributeError("self.clean_messages is empty.")
        
        splitted_cleaned_data = [x.split("\t") for x in self.cleaned_data]
        cleaned_frame = pd.DataFrame(splitted_cleaned_data)
        cleaned_frame.drop(index=[0], inplace=True)
    
        cleaned_frame.reset_index(inplace=True)
        cleaned_frame.drop(columns=df.columns[[0]], inplace=True)

        self.__cleaned_frame = cleaned_frame

    def modify_cleaned_frame(self) -> None:
        """Modify self.cleaned_frame.

        Raise
        -----
        AttributeError :
            if self.cleaned_frame is empty
        """
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

        Parameter
        ---------
        output_path : str

        Raise
        -----
        AttributeError :
            whether self.modified_cleaned_frame is empty
        """
        if self.modified_cleaned_frame.empty:
            raise AttributeError("self.modified_cleaned_frame is empty.")

        frame = self.modified_cleaned_frame.set_axis(["input", "output"], axis=1)
        pd.to_pickle(frame, output_path)

    def run_all(self, input_path: str, output_path: str) -> None:
        """Run all processes.

        Parameter
        ---------
        input_path : str
        output_path :str
        """
        self.read_text(input_path)
        self.clean_data()
        self.make_cleaned_frame()
        self.modify_cleaned_frame()
        self.save_as_pickle(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Line history preprocessor")

    # Add arguments: [https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0]
    parser.add_argument("-c", "--config_yaml_path", required=True, type=str)

    args = parser.parse_args()

    with open(args.config_yaml_path, "r") as yaml_f:
        config = yaml.safe_load(yaml_f)
    config_line = config["line"]
    config_initial = config_line["initial"]
    config_path = config_line["path"]

    line_preprcessor = LinePreProcessor(**config_initial)
    line_preprcessor.run_all(**config_path)
