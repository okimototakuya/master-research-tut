import datetime
import numpy as np


class IterAdd():

    def __init__(self, input_date_time):
        self.date_time = input_date_time

    def __iter__(self):
        for _ in range(5):
            self.date_time = self.date_time + datetime.timedelta(microseconds=np.random.binomial(1, 0.6)*100000)
            yield self.date_time.strftime('%M:%S.%f')


def main():
    date_time = datetime.datetime(2018, 12, 19, 14, minute=00, second=00, microsecond=0)
    iter_add = IterAdd(date_time)
    for i in iter_add:
        print(i)


if __name__ == '__main__':
    main()
