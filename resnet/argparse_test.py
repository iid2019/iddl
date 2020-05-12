r'''
Usage:
python argparse_test.py -m model_name
python argparse_test.py --model model_name
python argparse_test.py --model=model_name
'''

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help='The only one model you want to run.')
args = parser.parse_args()

print(args.model)
