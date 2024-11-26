from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import configparser
import pickle
from ast import literal_eval

import torch

if __name__ == "__main__":
    
    # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", default="./config.ini", help="Location to your global config file")
    args = vars(parser.parse_args())

    CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    print('CONFIG file being used: ', args["config"])
    CONFIG.read(args["config"])

    # file_path = CONFIG["preprocessing"]["pickle_path"]
    # with open(file_path, "rb") as f:
    #     dataset = pickle.load(f)

    train_val_test_split_ratio = literal_eval(CONFIG.get("preprocessing", 'train_val_test_split_ratio'))
    
    if len(train_val_test_split_ratio)==3:

        train_size = int(train_val_test_split_ratio[0] * len(dataset))
        validation_size = int(train_val_test_split_ratio[1] * len(dataset))
        test_size = len(dataset) - train_size - validation_size

        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, validation_size, test_size]
        )
        print(
            "sizes (train, val, test, total): ",
            len(train_dataset),
            len(validation_dataset),
            len(test_dataset),
            len(train_dataset) + len(validation_dataset) + len(test_dataset),
        )
        with open(file_path.replace(".pickle", "_train.pickle"), "ab") as f:
            pickle.dump(train_dataset, f)
        with open(file_path.replace(".pickle", "_validation.pickle"), "ab") as f:
            pickle.dump(validation_dataset, f)
        with open(file_path.replace(".pickle", "_test.pickle"), "ab") as f:
            pickle.dump(test_dataset, f)
    else:
        
        train_size = int(train_val_test_split_ratio[0] * len(dataset))
        validation_size = len(dataset) - train_size

        train_dataset, validation_dataset = torch.utils.data.random_split(
            dataset, [train_size, validation_size]
        )
        print(
            "sizes (train, val, total): ",
            len(train_dataset),
            len(validation_dataset),
            len(train_dataset) + len(validation_dataset),
        )
        with open(file_path.replace(".pickle", "_train.pickle"), "ab") as f:
            pickle.dump(train_dataset, f)
        with open(file_path.replace(".pickle", "_validation.pickle"), "ab") as f:
            pickle.dump(validation_dataset, f)
