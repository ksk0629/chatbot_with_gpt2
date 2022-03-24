import argparse
import pickle

from transformers import AutoTokenizer
import yaml


def make_training_data(basemodel: str, input_path: str, output_path: str) -> None:
    """Make and save training data to output_path using given basemodel from the file, which is specified by input_path.

    :param str basemodel: basemodel name
    :param str input_path: path to pickle data made through preprocessor
    :param str output_path: path to text data to train a gpt-2
    """
    tokenizer = AutoTokenizer.from_pretrained(basemodel)

    # Load input data
    with open(input_path, 'rb') as f:
        frame = pickle.load(f)

    all_data = []
    for x, y in zip(frame["input"], frame["output"]):
        input_tokens = tokenizer.tokenize(x)[:tokenizer.model_max_length]
        joined_input = "".join(input_tokens).replace('▁', '')

        output_tokens = tokenizer.tokenize(y)[:tokenizer.model_max_length]
        joined_output = "".join(output_tokens).replace('▁', '')

        data = "<s>" + joined_input + "[SEP]" + joined_output + "</s>"
        all_data.append(data)

    text = "".join(all_data)

    # Save data modified for training gpt-2
    with open(output_path, 'w') as output_file:
        output_file.write(text + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make training data for gpt-2")

    parser.add_argument("-c", "--config_yaml_path", required=False, type=str, default="model_config.yaml")
    args = parser.parse_args()

    with open(args.config_yaml_path, "r") as yaml_f:
        config = yaml.safe_load(yaml_f)
    config_general = config["general"]
    config_dataset = config["dataset"]

    make_training_data(**config_general, **config_dataset)
