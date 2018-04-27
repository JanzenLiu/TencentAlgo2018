import numpy as np
import statistics
import scipy.special as special
from tqdm import tqdm


np.random.seed(0)


class BayesianSmoothedClickrate:
    def __init__(self, alpha=1, beta=1, max_iter=1000, epsilon=1e-10,
                 use_moment=False, use_fixed_point=True):
        self.alpha = alpha  # initial alpha
        self.beta = beta  # initial beta
        self.max_iter = max_iter  # I strongly recommend a large `max_iter`, like 10000
        self.epsilon = epsilon
        self.use_moment = use_moment
        self.use_fixed_point = use_fixed_point

    @staticmethod
    def static_sample(alpha, beta, imps):
        """Generate simulated click counts with given alpha, beta and impression counts.

        Arguments
        ---------
        alpha: float
            Alpha used for the sampling of click rates.

        beta: float
            Beta used for the sampling of click rates.

        imps: array-like
            Array-like containing impression counts to generate click counts from.
            It can be specified as an array of two or even more dimensions, but I strong
            recommend you to pass an 1-D array.
        """
        imps = np.array(imps)
        clk_rates = np.random.beta(alpha, beta, imps.shape)  # generate click rate from beta distribution
        clks = np.round(imps * clk_rates)
        return clks, clk_rates

    @staticmethod
    def moment_solve(imps, clks):
        """Solve alpha and beta with moment estimation.

        Arguments
        ---------
        imps: array-like
            Array-like containing impression counts for each case.

        clks: array-like
            Array-like containing click counts for each case.
        """
        imps = np.array(imps)
        clks = np.array(clks)
        mask = (imps > 0)
        imps = imps[mask]
        clks = clks[mask]

        samples = clks / imps  # sample click rates calculated from impressions and clicks
        sample_mean = statistics.mean(samples)
        sample_var = statistics.variance(samples)
        shared_factor = (sample_mean * (1 - sample_mean) / sample_var - 1)
        alpha = shared_factor * sample_mean
        beta = shared_factor * (1 - sample_mean)
        return alpha, beta

    @staticmethod
    def iter_solve(imps, clks, alpha=1, beta=1, max_iter=1000, epsilon=1e-10, verbose=True):
        """Solve alpha and beta with repeated fixed-point iterations.

        I strongly recommend a larger max_iter, like 10000, in this
        competition. Or more generally, the more imbalanced the clicked
        and unclicked counts are, the larger the `max_iter` should be.
        However, it's just my guess according to some observations, I have
        no proof yet :)

        Arguments
        ---------
        imps: array-like
            Array-like containing impression counts for each case.

        clks: array-like
            Array-like containing click counts for each case.

        alpha: float
            Initial value of alpha.

        beta: float
            Initial value of beta.

        max_iter: int
            Maximum number of iteration. default 1000.

        epsilon: float
            Epsilon in the fixed-point iterations. It specifies the threshold
            of change in alpha and beta. The iterations will stop if both the change
            in alpha and beta is smaller than this value.

        verbose: boolean
            Whether to disable the progress bar of tqdm. You know some time when
            you are fitting thousands of bayesian model, too many progress bar output
            can be annoying.
        """
        imps = np.array(imps)
        clks = np.array(clks)
        mask = (imps > 0)
        imps = imps[mask]
        clks = clks[mask]
        for i in tqdm(range(max_iter), disable=(not verbose)):
            new_alpha, new_beta = BayesianSmoothedClickrate.fixed_point_iteration(imps, clks,
                                                                                  alpha, beta)
            if abs(new_alpha - alpha) < epsilon and abs(new_beta - beta) < epsilon:
                break
            alpha = new_alpha
            beta = new_beta
        return alpha, beta

    @staticmethod
    def fixed_point_iteration(imps, clks, alpha, beta):
        """Given alpha and beta, calculate updated values for them with impression and click counts.

        Adapted from: http://www.cnblogs.com/bentuwuying/p/6498370.html
        I optimized the code by replacing simple `for` loop with array operation, now it's more
        than 10 times faster :)

        Arguments
        ---------
        imps: array-like
            Array-like containing impression counts for each case.

        clks: array-like
            Array-like containing click counts for each case.

        alpha: float
            Initial value of alpha.

        beta: float
            Initial value of beta.

        Returns
        -------
        new_alpha: float
            Updated value of alpha.

        new_beta: float
            Updated value of beta.
        """
        assert len(imps) == len(clks)
        imps = np.array(imps).reshape((-1))
        clks = np.array(clks).reshape((-1))

        numerator_alpha = np.sum(special.digamma(clks + alpha) - special.digamma(alpha))
        numerator_beta = np.sum(special.digamma(imps - clks + beta) - special.digamma(beta))
        denominator = np.sum(special.digamma(imps + alpha + beta) - special.digamma(alpha + beta))
        new_alpha = alpha * (numerator_alpha / denominator)
        new_beta = beta * (numerator_beta / denominator)
        return new_alpha, new_beta

    @staticmethod
    def static_transform(imps, clks, alpha, beta):
        imps = np.array(imps)
        clks = np.array(clks)
        numerators = clks + alpha
        denominators = imps + alpha + beta
        return numerators / denominators

    @property
    def clickrate_expectation(self):
        return self.alpha / (self.alpha + self.beta)

    def sample(self, imps):
        return BayesianSmoothedClickrate.static_sample(self.alpha, self.beta, imps)

    def fit(self, imps, clks, verbose=True):
        """Fit the instance's alpha and beta with given impressions and clicks data.

        Arguments
        ---------
        imps: array-like
            Array-like containing impression counts for each case.

        clks: array-like
            Array-like containing click counts for each case.

        verbose: boolean
            Whether to disable the progress bar of tqdm. You know some time when
            you are fitting thousands of bayesian model, too many progress bar output
            can be annoying.
        """
        if self.use_moment:
            self.alpha, self.beta = BayesianSmoothedClickrate.moment_solve(imps, clks)
        if self.use_fixed_point:
            self.alpha, self.beta = BayesianSmoothedClickrate.iter_solve(imps=imps, clks=clks,
                                                                         alpha=self.alpha,
                                                                         beta=self.beta,
                                                                         max_iter=self.max_iter,
                                                                         epsilon=self.epsilon,
                                                                         verbose=verbose)
        return self

    def transform(self, imps, clks):
        return BayesianSmoothedClickrate.static_transform(imps, clks, self.alpha, self.beta)

    def fit_transform(self, imps, clks):
        return self.fit(imps, clks).transform(imps, clks)