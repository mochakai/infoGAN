import sys
import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt


def show_result(res_dict):
    fig, ax = plt.subplots()

    ax.plot(range(len(res_dict['loss_d'])), res_dict['loss_d'], label='loss_d')
    ax.plot(range(len(res_dict['loss_g'])), res_dict['loss_g'], label='loss_g')
    ax.plot(range(len(res_dict['loss_q'])), res_dict['loss_q'], label='loss_q')
    
    ax.set_title("LOSS of generator / discriminator / Q")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(range(len(res_dict['real_score'])), res_dict['real_score'], label='real_score')
    ax.plot(range(len(res_dict['fake_score_before'])), res_dict['fake_score_before'], label='fake_score_before')
    ax.plot(range(len(res_dict['fake_score_after'])), res_dict['fake_score_after'], label='fake_score_after')
    
    ax.set_title("score")
    ax.legend()
    ax.grid()
    plt.show()


def main(args):
    source = {}
    with open(args.file_name, 'r') as f:
        source = json.load(f)
    show_result(source)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("file_name", help="your json file name", type=str, default='')
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())

    sys.stdout.flush()
    sys.exit()