import os
import argparse
from test import test

def main():
    """
    The main function of your running scripts. 
    """
    # default data folder
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    args = parser.parse_args()

    test(args.input, args.output)

if __name__ == "__main__":
	main()