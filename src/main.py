# imports
import warnings
# warnings.filterwarnings("ignore")


# Classes
from CNN import CNN
from LSTM import LSTM


def main():
    print("here")
    leNet = CNN()

    leNet.__int__(inputShape=(10,10), outputShape=(1), _epoches=10, _batchSize=10)
    return


if __name__ == "__main__":
    main()
