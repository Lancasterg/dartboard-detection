import matplotlib.pyplot as plt


def main():
    tpr = [1, 1, 1, 1, 1]
    fpr = [1, 0.0603282, 0.00877532, 0.000152657, 2.60489e-06]

    for x in range(0, len(tpr)):
        plt.scatter(fpr[x], tpr[x], s=30, marker='x', label="Stage: {}".format(x))

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.ylim([0, 1.1])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

