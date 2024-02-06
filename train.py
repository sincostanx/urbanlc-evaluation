import yaml
from jsonargparse import ArgumentParser
import urbanlc

def main(configs, sensor):
    assert sensor in ["MSS", "TM", "OLITIRS"]
    params = {
        "architecture": configs["architecture"],
        "model_params": configs["model_params"],
        "device": configs["device"],
        "save_path": configs["save_path"],
    }
    classifier = getattr(urbanlc.model, f"{sensor}DeepLearning")(**params)
    classifier.train(**configs["training"])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True ,help='path to config file for training model')
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        configs = yaml.safe_load(f)

    sensor = configs["training"]["dataloader_params"]["sensor_type"]
    main(configs, sensor)