import argparse
import pickle
import yaml

from transformers import AutoTokenizer


def make_training_data(basemodel: str, input_path: str, output_path: str) -> None:
    """Make and save training data to output_path using given basemodel from the file, which is specified by input_path.

    Parameters
    ----------
    basemodel : str
        basemodel name
    input_path : str
    output_path : str
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
    parser = argparse.ArgumentParser(description="gpt-2 training data maker")

    # Add arguments: [https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0]
    parser.add_argument("-c", "--config_yaml_path", required=False, type=str, default="model_config.yaml")

    args = parser.parse_args()

    with open(args.config_yaml_path, "r") as yaml_f:
        config = yaml.safe_load(yaml_f)
    config_general = config["general"]
    config_dataset = config["dataset"]

    make_training_data(**config_general, **config_dataset)
