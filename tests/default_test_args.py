import argparse


def create_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", type=str, help="Name of algorithm to use", default="Rounding")
    parser.add_argument("-f", "--function", type=str, help="Name of test function", default="Branin")
    parser.add_argument("-k", "--kernel", type=str, help="Name of the kernel to test", default="LV")

    return parser
