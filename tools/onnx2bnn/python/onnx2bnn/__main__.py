import argparse

import onnx
import onnx2bnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model', help='Input ONNX model')
    parser.add_argument('output_model', help='Output dabnn model')
    parser.add_argument('level', nargs='?',
                        help='Level of onnx2bnn (possible value: moderate(default), strict and aggressive)', default='moderate')
    args = parser.parse_args()
    onnx2bnn.convert(args.input_model, args.output_model, args.level)


if __name__ == '__main__':
    main()
