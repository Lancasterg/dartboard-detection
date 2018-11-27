import matplotlib.pyplot as plt


def main():
    tpr = [1, 1, 1, 1, 1]
    fpr = [1, 0.0603282, 0.00877532, 0.000152657, 2.60489e-06]
    stages = [0, 1, 2, 3, 4]

    plt.plot(stages, tpr, label="True postivie rate")
    plt.plot(stages, fpr, label="False positive rate")
    plt.xlabel("Training Stage")
    plt.ylabel("True Positive / False Positive Rate")
    plt.ylim([-.01, 1.1])
    plt.xlim([-.01, 1.1])
    plt.xticks([0,1,2,3,4])
    plt.legend( loc = 'center right')
    plt.show()


if __name__ == '__main__':
    main()

