from colorama import Fore, Back, Style
import numpy as np


def print_banner():
    filepath = './banner.txt'
    with open(filepath) as fp:
        line = fp.readline()
        while line:
            print(Fore.GREEN + "{}".format(line))
            line = fp.readline()

    print("\n" + Fore.CYAN + "--Welcome to Compress Bot--")


def do_preprocessing(X):
    pass


def main():
    print_banner()


if __name__ == "__main__":
    main()
