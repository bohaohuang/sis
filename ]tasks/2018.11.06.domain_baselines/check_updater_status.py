import numpy as np


class bayes_update:
    def __init__(self):
        self.m = 0
        self.mean = 0
        self.var = 0

    def update(self, d):
        n = d.shape[0]
        mu_n = np.mean(d)
        sig_n = np.var(d)
        factor_m = self.m / (self.m + n)
        factor_n = 1 - factor_m

        mean_update = factor_m * self.mean + factor_n * mu_n
        self.var = factor_m * (self.var + self.mean ** 2) + factor_n * (sig_n + mu_n ** 2) - mean_update ** 2
        self.mean = mean_update

        self.m += n

        return np.array([self.mean, self.var])


class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.transpose(np.atleast_2d(data))
            self.mean = data.mean(axis=0)
            self.std  = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.transpose(np.atleast_2d(data))
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd  = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = np.sqrt(self.std)

            self.nobservations += n


if __name__ == '__main__':
    n_sample = 10000000
    bs = 10

    mean = 666
    std = 666
    d = np.random.normal(mean, std, n_sample)

    d = np.sort(d)

    up = StatsRecorder()
    up_mine = bayes_update()

    for i in range(0, n_sample, bs):
        up.update(d[i:i+bs])
        up_mine.update(d[i:i+bs])

    print('Mean = {:.3f}, Std = {:.3f}'.format(up.mean[0], up.std[0]))
    print('Mine Mean = {:.3f}, Std = {:.3f}'.format(up_mine.mean, np.sqrt(up_mine.var)))
