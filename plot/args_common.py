import argparse


def get_common_args():
    parser = argparse.ArgumentParser(description='A test program.')
    parser.add_argument("-fp", "--file_path", help="Path of CSV for KPI results", type=str)

    return parser
