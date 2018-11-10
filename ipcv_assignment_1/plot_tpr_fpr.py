import matplotlib.pyplot as plt
import csv


def main():
    with open('train_results.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            print(row)


if __name__ == '__main__':
    main()

