import os
import sys
import argparse

from palette_extract import Palette_Extract

parser = argparse.ArgumentParser(description='Generate a palette of the most used colors in a miniature')

parser.add_argument(
    'Image',
    metavar='img',
    type=str,
    help='The path to the image file of the miniature.'
)

parser.add_argument(
    '-p',
    '--palette',
    action='store',
    help='Name of the output file for the palette.'
)

parser.add_argument(
    '-t',
    '--test',
    action='store_true',
    help='Print intermediate image files, for testing purposes.'
)

args = parser.parse_args()

input_img = args.Image
output_palette = args.palette
test_files = args.test

if not os.path.exists(input_img):
    print('The file specified does not exist')
    sys.exit()

Palette_Extract(input_img, output_palette, test_files)