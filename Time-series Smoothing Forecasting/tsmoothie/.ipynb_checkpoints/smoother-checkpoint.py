'''
Define Smoother classes.
'''

import numpy as np
from scipy.signal import fftconvolve
import simdkalman

from .utils_class import LinearRegression
from .utils_func import (create_windows, sigma_interval, kalman_interval,
                         confidence_interval, prediction_interval)
from .utils_func import (_check_noise_dict, _check_knots, _check_weights,
                         _check_data, _check_data_nan, _check_output)
from .regression_basis import (polynomial, linear_spline, cubic_spline, natural_cubic_spline,
                               gaussian_kernel, binner, lowess)


_interval_types = {
    'KalmanSmoother':
        ['sigma_interval', 'kalman_interval'],
    'PolynomialSmoother':
        ['sigma_interval', 'confidence_interval', 'prediction_interval'],
    'SplineSmoother':
        ['sigma_interval', 'confidence_interval', 'prediction_interval'],
    'GaussianSmoother':
        ['sigma_interval', 'confidence_interval', 'prediction_interval'],
    'BinnerSmoother':
        ['sigma_interval', 'confidence_interval', 'prediction_interval'],
    'LowessSmoother':
        ['sigma_interval', 'confidence_interval', 'prediction_interval'],
    'ExponentialSmoother':
        ['sigma_interval'],
    'ConvolutionSmoother':
        ['sigma_interval'],
    'DecomposeSmoother':
        ['sigma_interval'],
    'SpectralSmoother':
        ['sigma_interval']
}


class _BaseSmoother:
    """Base class to build each Smoother.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self, copy=True):
        self.copy = copy

    def __repr__(self):
        return "<tsmoothie.smoother.{}>".format(self.__class__.__name__)

    def __str__(self):
        return "<tsmoothie.smoother.{}>".format(self.__class__.__name__)

    def _store_results(self, smooth_data, **objtosave):
        """Private method to store results."""

        self.smooth_data = smooth_data

        if self.copy:
            for name, obj in objtosave.items():
                setattr(self, str(name), obj)

    def get_intervals(self, interval_type, confidence=0.05, n_sigma=2):
        """Obtain intervals from the smoothed timeseries.
        Take care to set copy=True when defining the smoother.

        Supported interval types are:
        1) 'sigma_interval';
        2) 'confidence_interval';
        3) 'prediction_interval';
        4) 'kalman_interval'.

        Each Smooter supports different interval types:
        - 'KalmanSmoother' => (1,4);
        - 'PolynomialSmoother' => (1,2,3);
        - 'SplineSmoother' => (1,2,3);
        - 'GaussianSmoother' => (1,2,3);
        - 'BinnerSmoother' => (1,2,3);
        - 'LowessSmoother' => (1,2,3);
        - 'ExponentialSmoother' => (1);
        - 'ConvolutionSmoother' => (1);
        - 'DecomposeSmoother' => (1);
        - 'SpectralSmoother' => (1);
        - 'WindowWrapper' => depends on the Smoother received.

        Parameters
        ----------
        interval_type : str
            Type of interval used to produce the lower and upper bands.

        confidence : float, default=0.05
            Effective only for 'confidence_interval', 'prediction_interval'
            or 'kalman_interval'.
            The significance level for the intervals calculated as
            (1-confidence).

        n_sigma : int, default=2
            Effective only for 'sigma_interval'.
            How many standard deviations, calculated on residuals of the
            smoothing operation, are used to obtain the intervals.

        Returns
        -------
        low : array of shape (series, timesteps)
            Lower bands.

        up : array of shape (series, timesteps)
            Upper bands.
        """

        if self.__class__.__name__ == 'WindowWrapper':

            if not hasattr(self.Smoother, 'data'):
                raise ValueError(
                    "Pass some data to the smoother before computing intervals, "
                    "setting copy=True")

            if interval_type not in _interval_types[self.Smoother.__class__.__name__]:
                raise ValueError(
                    "'{}' is not a supported interval type for this smoother. "
                    "Supported types are {}".format(
                        interval_type, _interval_types[self.Smoother.__class__.__name__]))

            if interval_type == 'sigma_interval':
                low, up = sigma_interval(
                    self.Smoother.data, self.Smoother.smooth_data, n_sigma)

            elif interval_type == 'kalman_interval':
                low, up = kalman_interval(
                    self.Smoother.data, self.Smoother.smooth_data,
                    self.Smoother.cov, confidence)

            elif (interval_type == 'confidence_interval' or
                  interval_type == 'prediction_interval'):
                interval_f = eval(interval_type)
                low, up = interval_f(
                    self.Smoother.data, self.Smoother.smooth_data,
                    self.Smoother.X, confidence)

        else:

            if not hasattr(self, 'data'):
                raise ValueError(
                    "Pass some data to the smoother before computing intervals, "
                    "setting copy=True")

            if interval_type not in _interval_types[self.__class__.__name__]:
                raise ValueError(
                    "'{}' is not a supported interval type for this smoother. "
                    "Supported types are {}".format(
                        interval_type, _interval_types[self.__class__.__name__]))

            if interval_type == 'sigma_interval':
                low, up = sigma_interval(self.data, self.smooth_data, n_sigma)

            elif interval_type == 'kalman_interval':
                low, up = kalman_interval(
                    self.data, self.smooth_data, self.cov, confidence)

            elif (interval_type == 'confidence_interval' or
                  interval_type == 'prediction_interval'):
                interval_f = eval(interval_type)
                low, up = interval_f(
                    self.data, self.smooth_data, self.X, confidence)

        return low, up


class ExponentialSmoother(_BaseSmoother):
    """ExponentialSmoother operates convolutions of fixed dimensions
    on the series using a weighted windows. The weights are the same
    for all windows and are computed using an exponential decay.
    The most recent observations are most important than the past ones.
    This is imposed choosing a parameter (alpha).
    No padded is provided in order to not alter the results at the edges.
    For this reason, this technique doesn't operate smoothing until
    the observations at position window_len.

    The ExponentialSmoother automatically vectorizes, in an efficient way,
    the desired smoothing operation on all the series received.

    Parameters
    ----------
    window_len : int
        Greater than equal to 1. The length of the window used to compute
        the exponential smoothing.

    alpha : float
        Between 0 and 1. (1-alpha) provides the importance of the past
        obsevations when computing the smoothing.

    copy : bool, default=True
        If True, the raw data received by the smoother and the smoothed
        results can be accessed using 'data' and 'smooth_data' attributes.
        This is useful to calculate the intervals. If set to False the
        interval calculation is disabled. In order to save memory, set it to
        False if you are interested only in the smoothed results.

    Attributes
    ----------
    smooth_data : array of shape (series, timesteps-window_len)
        Smoothed data derived from the smoothing operation.
        It has the same shape of the raw data received without the first
        observations until window_len. It is accessible after computhing
        smoothing, otherwise None is returned.

    data : array of shape (series, timesteps-window_len)
        Raw data received by the smoother. It is accessible with 'copy'=True
        and after computhing smoothing, otherwise None is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *
    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=10, timesteps=200,
    ...                       process_noise=10, measure_noise=30)
    >>> smoother = ExponentialSmoother(window_len=20, alpha=0.3)
    >>> smoother.smooth(data)
    >>> low, up = smoother.get_intervals('sigma_interval')
    """

    def __init__(self, window_len, alpha, copy=True):
        self.window_len = window_len
        self.alpha = alpha
        self.copy = copy

    def smooth(self, data):
        """Smooth timeseries.

        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,)
               for single timeseries
            Timeseries to smooth. The data are assumed to be in increasing
            time order in each timeseries.

        Returns
        -------
        self : returns an instance of self
        """

        if self.window_len < 1:
            raise ValueError("window_len must be >= 1")

        if self.alpha > 1 or self.alpha < 0:
            raise ValueError("alpha must be in the range [0,1]")

        data = _check_data(data)

        if self.window_len >= data.shape[0]:
            raise ValueError(
                "window_len must be < than timesteps dimension "
                "of the data received")

        w = np.power((1 - self.alpha), np.arange(self.window_len))

        if data.ndim == 2:
            w = np.repeat([w / w.sum()], data.shape[1], axis=0).T
        else:
            w = w / w.sum()

        smooth = fftconvolve(w, data, mode='full', axes=0)
        smooth = smooth[self.window_len:data.shape[0]]
        data = data[self.window_len:data.shape[0]]

        smooth = _check_output(smooth)
        data = _check_output(data)

        self._store_results(smooth_data=smooth, data=data)

        return self


class ConvolutionSmoother(_BaseSmoother):
    """ConvolutionSmoother operates convolutions of fixed dimensions
    on the series using a weighted windows. The weights can assume
    different format but they are the same for all the windows and
    fixed for the whole procedure. The series are padded, reflecting themself,
    with a quantity equal to the window size in both ends to avoid loss of
    information.

    The ConvolutionSmoother automatically vectorizes, in an efficient way,
    the desired smoothing operation on all the series received.

    Parameters
    ----------
    window_len : int
        Greater than equal to 1. The length of the window used to compute
        the convolutions.

    window_type : str
        The type of the window used to compute the convolutions.
        Supported types are: 'ones', 'hanning', 'hamming', 'bartlett', 'blackman'.

    copy : bool, default=True
        If True, the raw data received by the smoother and the smoothed
        results can be accessed using 'data' and 'smooth_data' attributes.
        This is useful to calculate the intervals. If set to False the
        interval calculation is disabled. In order to save memory, set it to
        False if you are interested only in the smoothed results.

    Attributes
    ----------
    smooth_data : array of shape (series, timesteps)
        Smoothed data derived from the smoothing operation. It is accessible
        after computhing smoothing, otherwise None is returned.

    data : array of shape (series, timesteps)
        Raw data received by the smoother. It is accessible with 'copy'=True
        and  after computhing smoothing, otherwise None is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *
    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=10, timesteps=200,
    ...                       process_noise=10, measure_noise=30)
    >>> smoother = ConvolutionSmoother(window_len=10, window_type='ones')
    >>> smoother.smooth(data)
    >>> low, up = smoother.get_intervals('sigma_interval')
    """

    def __init__(self, window_len, window_type, copy=True):
        self.window_len = window_len
        self.window_type = window_type
        self.copy = copy

    def smooth(self, data):
        """Smooth timeseries.

        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,)
               for single timeseries
            Timeseries to smooth. The data are assumed to be in increasing
            time order in each timeseries.

        Returns
        -------
        self : returns an instance of self
        """

        window_types = ['ones', 'hanning', 'hamming', 'bartlett', 'blackman']

        if self.window_type not in window_types:
            raise ValueError(
                "'{}' is not a supported window type. "
                "Supported types are {}".format(self.window_type, window_types))

        if self.window_len < 1:
            raise ValueError("window_len must be >= 1")

        data = _check_data(data)

        if self.window_len % 2 == 0:
            window_len = int(self.window_len + 1)
        else:
            window_len = self.window_len

        if self.window_type == 'ones':
            w = np.ones(window_len)
        else:
            w = eval('np.' + self.window_type + '(window_len)')

        if data.ndim == 2:
            pad_data = np.pad(
                data, ((window_len, window_len), (0, 0)), mode='symmetric')
            w = np.repeat([w / w.sum()], pad_data.shape[1], axis=0).T
        else:
            pad_data = np.pad(data, window_len, mode='symmetric')
            w = w / w.sum()

        smooth = fftconvolve(w, pad_data, mode='valid', axes=0)
        smooth = smooth[(window_len // 2 + 1):-(window_len // 2 + 1)]

        smooth = _check_output(smooth)
        data = _check_output(data)

        self._store_results(smooth_data=smooth, data=data)

        return self


class SpectralSmoother(_BaseSmoother):
    """SpectralSmoother smoothes the timeseries applying a Fourier
    Transform. It maintains the most important frequencies, suppressing
    the others in the Fourier domain. This results in a smoother curves
    when returning to a real domain.

    The SpectralSmoother automatically vectorizes, in an efficient way,
    the desired smoothing operation on all the series passed.

    Parameters
    ----------
    smooth_fraction : float
        Between 0 and 1. The smoothing strength. A lower value of
        smooth_fraction will result in a smoother curve. It's the proportion
        of frequencies used in the discrete Fourier Transform to smooth
        the curve.

    pad_len : int
        Greater than equal to 1. The length of the padding used at each
        timeseries edge to center the series and obtain better smoothings.

    copy : bool, default=True
        If True, the raw data received by the smoother and the smoothed
        results can be accessed using 'data' and 'smooth_data' attributes.
        This is useful to calculate the intervals. If set to False the
        interval calculation is disabled. In order to save memory, set it to
        False if you are interested only in the smoothed results.

    Attributes
    ----------
    smooth_data : array of shape (series, timesteps)
        Smoothed data derived from the smoothing operation. It is accessible
        after computhing smoothing, otherwise None is returned.

    data : array of shape (series, timesteps)
        Raw data received by the smoother. It is accessible with 'copy'=True
        and  after computhing smoothing, otherwise None is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils_func import sim_seasonal_data
    >>> from tsmoothie.smoother import *
    >>> np.random.seed(33)
    >>> data = sim_seasonal_data(n_series=3, timesteps=200,
    ...                          freq=24, measure_noise=15)
    >>> smoother = SpectralSmoother(smooth_fraction=0.2, pad_len=20)
    >>> smoother.smooth(data)
    >>> low, up = smoother.get_intervals('sigma_interval')
    """

    def __init__(self, smooth_fraction, pad_len, copy=True):
        self.smooth_fraction = smooth_fraction
        self.pad_len = pad_len
        self.copy = copy

    def smooth(self, data):
        """Smooth timeseries.

        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,)
               for single timeseries
            Timeseries to smooth. The data are assumed to be in increasing
            time order in each timeseries.

        Returns
        -------
        self : returns an instance of self
        """

        if self.smooth_fraction >= 1 or self.smooth_fraction <= 0:
            raise ValueError("smooth_fraction must be in the range (0,1)")

        if self.pad_len < 1:
            raise ValueError("pad_len must be >= 1")

        data = _check_data(data)

        if data.ndim == 2:
            pad_data = np.pad(
                data, ((self.pad_len, self.pad_len), (0, 0)),
                mode='symmetric')
        else:
            pad_data = np.pad(data, self.pad_len, mode='symmetric')

        rfft = np.fft.rfft(pad_data, axis=0)
        n_coeff = int(rfft.shape[0] * self.smooth_fraction)

        if rfft.ndim == 2:
            rfft[n_coeff:, :] = 0
        else:
            rfft[n_coeff:] = 0

        if data.shape[0] % 2 > 0:
            n = 2 * rfft.shape[0] - 1
        else:
            n = 2 * (rfft.shape[0] - 1)

        smooth = np.fft.irfft(rfft, n=n, axis=0)
        smooth = smooth[self.pad_len:-self.pad_len]

        smooth = _check_output(smooth)
        data = _check_output(data)

        self._store_results(smooth_data=smooth, data=data)

        return self


class PolynomialSmoother(_BaseSmoother):
    """PolynomialSmoother smoothes the timeseries applying a linear
    regression on an ad-hoc basis expansion.
    The input space, used to build the basis expansion, consists in
    a single continuos increasing sequence.

    The PolynomialSmoother automatically vectorizes, in an efficient way,
    the desired smoothing operation on all the series received.

    Parameters
    ----------
    degree : int
        The polynomial order used to build the basis.

    copy : bool, default=True
        If True, the raw data received by the smoother and the smoothed
        results can be accessed using 'data' and 'smooth_data' attributes.
        This is useful to calculate the intervals. If set to False the
        interval calculation is disabled. In order to save memory, set it to
        False if you are interested only in the smoothed results.

    Attributes
    ----------
    smooth_data : array of shape (series, timesteps)
        Smoothed data derived from the smoothing operation. It is accessible
        after computhing smoothing, otherwise None is returned.

    data : array of shape (series, timesteps)
        Raw data received by the smoother. It is accessible with 'copy'=True
        and  after computhing smoothing, otherwise None is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *
    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=10, timesteps=200,
    ...                       process_noise=10, measure_noise=30)
    >>> smoother = PolynomialSmoother(degree=6)
    >>> smoother.smooth(data)
    >>> low, up = smoother.get_intervals('prediction_interval')
    """

    def __init__(self, degree, copy=True):
        self.degree = degree
        self.copy = copy

    def smooth(self, data, weights=None):
        """Smooth timeseries.

        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,)
               for single timeseries
            Timeseries to smooth. The data are assumed to be in increasing
            time order in each timeseries.

        weights : array-like of shape (timesteps,), default=None
            Individual weights for each timestep. In case of multidimesional
            timeseries, the same weights are used for all the timeseries.

        Returns
        -------
        self : returns an instance of self
        """

        if self.degree < 1:
            raise ValueError("degree must be > 0")

        data = _check_data(data)
        basis_len = data.shape[0]
        weights = _check_weights(weights, basis_len)

        X_base = polynomial(self.degree, basis_len)

        lr = LinearRegression(fit_intercept=True)
        lr.fit(X_base, data, sample_weight=weights)

        smooth = lr.predict(X_base)

        smooth = _check_output(smooth)
        data = _check_output(data)

        self._store_results(smooth_data=smooth, X=X_base, data=data)

        return self


class SplineSmoother(_BaseSmoother):
    """SplineSmoother smoothes the timeseries applying a linear regression
    on an ad-hoc basis expansion. Three types of spline smoothing are
    available: 'linear spline', 'cubic spline', 'natural cubic spline'.
    In all of the available methods, the input space consists in a single
    continuos increasing sequence.

    Two possibilities are available:
    - smooth the timeseries in equal intervals, where the number of
      intervals is a user defined parameter (n_knots);
    - smooth the timeseries in custom length intervals, where the interval
      positions are defined by the user as normalize points (knots).
    The two methods are exclusive: the usage of n_knots makes not effective
    the usage of knots and vice-versa.

    The SplineSmoother automatically vectorizes, in an efficient way,
    the desired smoothing operation on all the series received.

    Parameters
    ----------
    spline_type : str
        Type of spline smoother to operate. Supported types are 'linear_spline',
        'cubic_spline' or 'natural_cubic_spline'.

    n_knots : int
        Between 1 and timesteps for 'linear_spline' and 'natural_cubic_spline'.
        Between 3 and timesteps for 'natural_cubic_spline'.
        Number of equal intervals used to divide the input space and smooth
        the timeseries. A lower value of n_knots will result in a smoother curve.

    knots : array-like of shape (n_knots,), default=None
        With length of at least 1 for 'linear_spline' and 'natural_cubic_spline'.
        With length of at least 3 for 'natural_cubic_spline'.
        Normalized points in the range [0,1] that specify in which sections
        divide the input space. A lower number of knots will result in a
        smoother curve.

    copy : bool, default=True
        If True, the raw data received by the smoother and the smoothed
        results can be accessed using 'data' and 'smooth_data' attributes.
        This is useful to calculate the intervals. If set to False the
        interval calculation is disabled. In order to save memory, set it to
        False if you are interested only in the smoothed results.

    Attributes
    ----------
    smooth_data : array of shape (series, timesteps)
        Smoothed data derived from the smoothing operation. It is accessible
        after computhing smoothing, otherwise None is returned.

    data : array of shape (series, timesteps)
        Raw data received by the smoother. It is accessible with 'copy'=True
        and  after computhing smoothing, otherwise None is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *
    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=10, timesteps=200,
    ...                       process_noise=10, measure_noise=30)
    >>> smoother = SplineSmoother(n_knots=6, spline_type='natural_cubic_spline')
    >>> smoother.smooth(data)
    >>> low, up = smoother.get_intervals('prediction_interval')
    """

    def __init__(self, spline_type, n_knots, knots=None, copy=True):
        self.spline_type = spline_type
        self.n_knots = n_knots
        self.knots = knots
        self.copy = copy

    def smooth(self, data, weights=None):
        """Smooth timeseries.

        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,)
               for single timeseries
            Timeseries to smooth. The data are assumed to be in increasing
            time order in each timeseries.

        weights : array-like of shape (timesteps,), default=None
            Individual weights for each timestep. In case of multidimesional
            timeseries, the same weights are used for all the timeseries.

        Returns
        -------
        self : returns an instance of self
        """

        spline_types = {'linear_spline': 1, 'cubic_spline': 1,
                        'natural_cubic_spline': 3}

        if self.spline_type not in spline_types:
            raise ValueError("'{}' is not a supported spline type. "
                             "Supported types are {}".format(
                self.spline_type, list(spline_types.keys())))

        data = _check_data(data)
        basis_len = data.shape[0]
        weights = _check_weights(weights, basis_len)

        if self.knots is not None:
            knots = _check_knots(
                self.knots, spline_types[self.spline_type])[1:-1] * basis_len

        else:
            if self.n_knots < spline_types[self.spline_type]:
                raise ValueError(
                    "'{}' requires n_knots >= {}".format(
                        self.spline_type, spline_types[self.spline_type]))

            if self.n_knots > basis_len:
                raise ValueError(
                    "n_knots must be <= than timesteps dimension "
                    "of the data received")

            knots = np.linspace(0, basis_len, self.n_knots + 2)[1:-1]

        f = eval(self.spline_type)
        X_base = f(knots, basis_len)

        lr = LinearRegression(fit_intercept=True)
        lr.fit(X_base, data, sample_weight=weights)

        smooth = lr.predict(X_base)

        smooth = _check_output(smooth)
        data = _check_output(data)

        self._store_results(smooth_data=smooth, X=X_base, data=data)

        return self


class GaussianSmoother(_BaseSmoother):
    """GaussianSmoother smoothes the timeseries applying a linear
    regression on an ad-hoc basis expansion. The features created with
    this method are obtained applying a gaussian kernel centered to specified
    points of the input space.
    In timeseries domain, the input space consists in a single continuos
    increasing sequence.

    Two possibilities are available:
    - smooth the timeseries in equal intervals, where the number of
      intervals is a user defined parameter (n_knots);
    - smooth the timeseries in custom length intervals, where the interval
      positions are defined by the user as normalize points (knots).
    The two methods are exclusive: the usage of n_knots makes not effective
    the usage of knots and vice-versa.

    The GaussianSmoother automatically vectorizes, in an efficient way,
    the desired smoothing operation on all the series received.

    Parameters
    ----------
    sigma : float
        sigma in the gaussian kernel.

    n_knots : int
        Between 1 and timesteps. Number of equal intervals used to divide
        the input  space and smooth the timeseries. A lower value of n_knots
        will result in a smoother curve.

    knots : array-like of shape (n_knots,), default=None
        With length of at least 1. Normalized points in the range [0,1] that
        specify in which sections divide the input space. A lower number of
        knots will result in a smoother curve.

    copy : bool, default=True
        If True, the raw data received by the smoother and the smoothed
        results can be accessed using 'data' and 'smooth_data' attributes.
        This is useful to calculate the intervals. If set to False the
        interval calculation is disabled. In order to save memory, set it to
        False if you are interested only in the smoothed results.

    Attributes
    ----------
    smooth_data : array of shape (series, timesteps)
        Smoothed data derived from the smoothing operation. It is accessible
        after computhing smoothing, otherwise None is returned.

    data : array of shape (series, timesteps)
        Raw data received by the smoother. It is accessible with 'copy'=True
        and  after computhing smoothing, otherwise None is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *
    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=10, timesteps=200,
    ...                       process_noise=10, measure_noise=30)
    >>> smoother = GaussianSmoother(n_knots=6, sigma=0.1)
    >>> smoother.smooth(data)
    >>> low, up = smoother.get_intervals('prediction_interval')
    """

    def __init__(self, sigma, n_knots, knots=None, copy=True):
        self.sigma = sigma
        self.n_knots = n_knots
        self.knots = knots
        self.copy = copy

    def smooth(self, data, weights=None):
        """Smooth timeseries.

        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,)
               for single timeseries
            Timeseries to smooth. The data are assumed to be in increasing
            time order in each timeseries.

        weights : array-like of shape (timesteps,), default=None
            Individual weights for each timestep. In case of multidimesional
            timeseries, the same weights are used for all the timeseries.

        Returns
        -------
        self : returns an instance of self
        """

        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")

        data = _check_data(data)
        basis_len = data.shape[0]
        weights = _check_weights(weights, basis_len)

        if self.knots is not None:
            knots = _check_knots(self.knots, 1)[1:-1]

        else:
            if self.n_knots < 1:
                raise ValueError("n_knots must be > 0")

            if self.n_knots > basis_len:
                raise ValueError(
                    "n_knots must be <= than timesteps dimension "
                    "of the data received")

            knots = np.linspace(0, 1, self.n_knots + 2)[1:-1]

        X_base = gaussian_kernel(knots, self.sigma, basis_len)

        lr = LinearRegression(fit_intercept=True)
        lr.fit(X_base, data, sample_weight=weights)

        smooth = lr.predict(X_base)

        smooth = _check_output(smooth)
        data = _check_output(data)

        self._store_results(smooth_data=smooth, X=X_base, data=data)

        return self


class BinnerSmoother(_BaseSmoother):
    """BinnerSmoother smoothes the timeseries applying a linear regression
    on an ad-hoc basis expansion. The features created with this method
    are obtained binning the input space into intervals.
    An indicator feature is created for each bin, indicating where
    a given observation falls into.
    In timeseries domain, the input space consists in a single continuos
    increasing sequence.

    Two possibilities are available:
    - smooth the timeseries in equal intervals, where the number of
      intervals is a user defined parameter (n_knots);
    - smooth the timeseries in custom length intervals, where the interval
      positions are defined by the user as normalize points (knots).
    The two methods are exclusive: the usage of n_knots makes not effective
    the usage of knots and vice-versa.

    The BinnerSmoother automatically vectorizes, in an efficient way,
    the desired smoothing operation on all the series received.

    Parameters
    ----------
    n_knots : int
        Between 1 and timesteps. Number of equal intervals used to divide
        the input  space and smooth the timeseries. A lower value of n_knots
        will result in a smoother curve.

    knots : array-like of shape (n_knots,), default=None
        With length of at least 1. Normalized points in the range [0,1] that
        specify in which sections divide the input space. A lower number of
        knots will result in a smoother curve.

    copy : bool, default=True
        If True, the raw data received by the smoother and the smoothed
        results can be accessed using 'data' and 'smooth_data' attributes.
        This is useful to calculate the intervals. If set to False the
        interval calculation is disabled. In order to save memory, set it to
        False if you are interested only in the smoothed results.

    Attributes
    ----------
    smooth_data : array of shape (series, timesteps)
        Smoothed data derived from the smoothing operation. It is accessible
        after computhing smoothing, otherwise None is returned.

    data : array of shape (series, timesteps)
        Raw data received by the smoother. It is accessible with 'copy'=True
        and  after computhing smoothing, otherwise None is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *
    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=10, timesteps=200,
    ...                       process_noise=10, measure_noise=30)
    >>> smoother = BinnerSmoother(n_knots=6)
    >>> smoother.smooth(data)
    >>> low, up = smoother.get_intervals('prediction_interval')
    """

    def __init__(self, n_knots, knots=None, copy=True):
        self.n_knots = n_knots
        self.knots = knots
        self.copy = copy

    def smooth(self, data, weights=None):
        """Smooth timeseries.

        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,)
               for single timeseries
            Timeseries to smooth. The data are assumed to be in increasing
            time order in each timeseries.

        weights : array-like of shape (timesteps,), default=None
            Individual weights for each timestep. In case of multidimesional
            timeseries, the same weights are used for all the timeseries.

        Returns
        -------
        self : returns an instance of self
        """

        data = _check_data(data)
        basis_len = data.shape[0]
        weights = _check_weights(weights, basis_len)

        if self.knots is not None:
            knots = _check_knots(self.knots, 1)[1:-1] * basis_len

        else:
            if self.n_knots < 1:
                raise ValueError("n_knots must be > 0")

            if self.n_knots > basis_len:
                raise ValueError(
                    "n_knots must be <= than timesteps dimension "
                    "of the data received")

            knots = np.linspace(0, basis_len, self.n_knots + 2)[1:-1]

        X_base = binner(knots, basis_len)

        lr = LinearRegression(fit_intercept=True)
        lr.fit(X_base, data, sample_weight=weights)

        smooth = lr.predict(X_base)

        smooth = _check_output(smooth)
        data = _check_output(data)

        self._store_results(smooth_data=smooth, X=X_base, data=data)

        return self


class LowessSmoother(_BaseSmoother):
    """LowessSmoother uses LOWESS (locally-weighted scatterplot smoothing)
    to smooth the timeseries. This smoothing technique is a non-parametric
    regression method that essentially fit a unique linear regression
    for every data point by including nearby data points to estimate
    the slope and intercept. The presented method is robust because it
    performs residual-based reweightings simply specifing the number of
    iterations to operate.

    The LowessSmoother automatically vectorizes, in an efficient way,
    the desired smoothing operation on all the series passed.

    Parameters
    ----------
    smooth_fraction : float
        Between 0 and 1. The smoothing span. A larger value of smooth_fraction
        will result in a smoother curve.

    iterations : int
        Between 1 and 6. The number of residual-based reweightings to perform.

    batch_size : int, default=None
        How many timeseries are smoothed simultaneously. This parameter is
        important because LowessSmoother is a memory greedy process. Setting
        it low, with big timeseries, helps to avoid MemoryError. By default
        None means that all the timeseries are smoothed simultaneously.

    copy : bool, default=True
        If True, the raw data received by the smoother and the smoothed
        results can be accessed using 'data' and 'smooth_data' attributes.
        This is useful to calculate the intervals. If set to False the
        interval calculation is disabled. In order to save memory, set it to
        False if you are interested only in the smoothed results.

    Attributes
    ----------
    smooth_data : array of shape (series, timesteps)
        Smoothed data derived from the smoothing operation. It is accessible
        after computhing smoothing, otherwise None is returned.

    data : array of shape (series, timesteps)
        Raw data received by the smoother. It is accessible with 'copy'=True
        and  after computhing smoothing, otherwise None is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *
    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=10, timesteps=200,
    ...                       process_noise=10, measure_noise=30)
    >>> smoother = LowessSmoother(smooth_fraction=0.3, iterations=1)
    >>> smoother.smooth(data)
    >>> low, up = smoother.get_intervals('prediction_interval')
    """

    def __init__(self, smooth_fraction, iterations=1,
                 batch_size=None, copy=True):
        self.smooth_fraction = smooth_fraction
        self.iterations = iterations
        self.batch_size = batch_size
        self.copy = copy

    def smooth(self, data):
        """Smooth timeseries.

        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,)
               for single timeseries
            Timeseries to smooth. The data are assumed to be in increasing
            time order in each timeseries.

        Returns
        -------
        self : returns an instance of self
        """

        if self.smooth_fraction >= 1 or self.smooth_fraction <= 0:
            raise ValueError("smooth_fraction must be in the range (0,1)")

        if self.iterations <= 0 or self.iterations > 6:
            raise ValueError("iterations must be in the range (0,6]")

        data = _check_data(data)
        if data.ndim == 1:
            data = data[:, None]
        timesteps, n_timeseries = data.shape

        if self.batch_size is not None:
            if self.batch_size <= 0 or self.batch_size > n_timeseries:
                raise ValueError("batch_size must be in the range (0,series]")

        X = np.arange(timesteps) / (timesteps - 1)
        w_init = lowess(self.smooth_fraction, timesteps)

        delta = np.ones_like(data)

        if self.batch_size is None:
            batches = [np.arange(0, n_timeseries)]
        else:
            batches = np.split(
                np.arange(0, n_timeseries),
                np.arange(self.batch_size, n_timeseries + self.batch_size - 1,
                          self.batch_size))
        smooth = np.empty_like(data)

        for iteration in range(self.iterations):

            for B in batches:

                try:
                    w = delta[:, None, B] * w_init[..., None]
                    # (timesteps, timesteps, n_series)
                    wy = w * data[:, None, B]
                    # (timesteps, timesteps, n_series)
                    wyx = wy * X[:, None, None]
                    # (timesteps, timesteps, n_series)
                    wx = w * X[:, None, None]
                    # (timesteps, timesteps, n_series)
                    wxx = wx * X[:, None, None]
                    # (timesteps, timesteps, n_series)

                    b = np.array([wy.sum(axis=0), wyx.sum(axis=0)]).T
                    # (n_series, timesteps, 2)
                    A = np.array([[w.sum(axis=0), wx.sum(axis=0)],
                                  [wx.sum(axis=0), wxx.sum(axis=0)]])
                    # (2, 2, timesteps, n_series)

                    XtX = (A.transpose(1, 0, 2, 3)[None, ...] * A[:, None, ...]).sum(2)
                    # (2, 2, timesteps, n_series)
                    XtX = np.linalg.pinv(XtX.transpose(3, 2, 0, 1))
                    # (n_series, timesteps, 2, 2)
                    XtXXt = (XtX[..., None] * A.transpose(3, 2, 1, 0)[..., None, :]).sum(2)
                    # (n_series, timesteps, 2, 2)
                    betas = np.squeeze(XtXXt @ b[..., None], -1)
                    # (n_series, timesteps, 2)

                    smooth[:, B] = (betas[..., 0] + betas[..., 1] * X).T
                    # (timesteps, n_series)

                    residuals = data[:, B] - smooth[:, B]
                    s = np.median(np.abs(residuals), axis=0).clip(1e-5)
                    delta[:, B] = (residuals / (6.0 * s)).clip(-1, 1)
                    delta[:, B] = np.square(1 - np.square(delta[:, B]))

                except MemoryError:
                    raise StopIteration(
                        "Reduce the batch_size provided in order to not encounter "
                        "memory errors. Provided batch_size is {}. By default batch_size "
                        "is set to None. This means that all the timeseries "
                        "passed are smoothed simultaneously".format(self.batch_size))

        smooth = _check_output(smooth)
        data = _check_output(data)

        self._store_results(smooth_data=smooth, X=X * (timesteps - 1), data=data)

        return self


class DecomposeSmoother(_BaseSmoother):
    """DecomposeSmoother smoothes the timeseries applying a standard
    seasonal decomposition. The seasonal decomposition can be carried out
    using different smoothing techniques available in tsmoothie.

    The DecomposeSmoother automatically vectorizes, in an efficient way,
    the desired smoothing operation on all the series received.

    Parameters
    ----------
    smooth_type : str
        The type of smoothing used to compute the seasonal decomposition.
        Supported types are: 'convolution', 'lowess', 'natural_cubic_spline'.

    periods : list
        List of seasonal periods of the timeseries. Multiple periods are
        allowed. Each period must be an integer reater than 0.

    method : str, default='additive'
        Type of seasonal component.
        Supported types are: 'additive', 'multiplicative'.

    **smoothargs : Smoothing arguments
        The same accepted by the smoother referring to smooth_type.

    copy : bool, default=True
        If True, the raw data received by the smoother and the smoothed
        results can be accessed using 'data' and 'smooth_data' attributes.
        This is useful to calculate the intervals. If set to False the
        interval calculation is disabled. In order to save memory, set it to
        False if you are interested only in the smoothed results.

    Attributes
    ----------
    smooth_data : array of shape (series, timesteps)
        Smoothed data derived from the smoothing operation. It is accessible
        after computhing smoothing, otherwise None is returned.

    data : array of shape (series, timesteps)
        Raw data received by the smoother. It is accessible with 'copy'=True
        and  after computhing smoothing, otherwise None is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils_func import sim_seasonal_data
    >>> from tsmoothie.smoother import *
    >>> np.random.seed(33)
    >>> data = sim_seasonal_data(n_series=3, timesteps=300,
    ...                          freq=24, measure_noise=30)
    >>> smoother = DecomposeSmoother(smooth_type='convolution', periods=24,
    ...                              window_len=30, window_type='ones')
    >>> smoother.smooth(data)
    >>> low, up = smoother.get_intervals('sigma_interval')
    """

    def __init__(self, smooth_type, periods, method='additive', copy=True,
                 **smoothargs):
        self.smooth_type = smooth_type
        self.periods = periods
        self.method = method
        self.copy = copy
        self.smoothargs = smoothargs

    def smooth(self, data):
        """Smooth timeseries.

        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,)
               for single timeseries
            Timeseries to smooth. The data are assumed to be in increasing
            time order in each timeseries.

        Returns
        -------
        self : returns an instance of self
        """

        smooth_types = ['convolution', 'lowess', 'natural_cubic_spline']
        methods = ['additive', 'multiplicative']

        if self.smooth_type not in smooth_types:
            raise ValueError(
                "'{}' is not a supported smooth type. "
                "Supported types are {}".format(self.smooth_type, smooth_types))

        if self.method not in methods:
            raise ValueError("'{}' is not a supported method type. "
                             "Supported types are {}".format(
                self.method, methods))

        if not isinstance(self.periods, list):
            periods = [self.periods]
        else:
            periods = self.periods

        for p in periods:
            if p <= 0 or not isinstance(p, int):
                raise ValueError("periods must a list containing int > 0")

        if self.smooth_type == 'convolution':
            smoother = ConvolutionSmoother(copy=True, **self.smoothargs)
            smoother.smooth(data)
        elif self.smooth_type == 'lowess':
            smoother = LowessSmoother(copy=True, **self.smoothargs)
            smoother.smooth(data)
        elif self.smooth_type == 'natural_cubic_spline':
            smoother = SplineSmoother(copy=True, spline_type=self.smooth_type,
                                      **self.smoothargs)
            smoother.smooth(data)

        if self.method == 'additive':
            detrended = smoother.data - smoother.smooth_data
        else:
            detrended = smoother.data / smoother.smooth_data

        period_averages = []
        for p in periods:
            period_averages.append(
                np.array([np.mean(detrended[:, i::p], axis=1)
                          for i in range(p)]).T)

        if self.method == 'additive':
            period_averages = [p_a - np.mean(p_a, axis=1, keepdims=True)
                               for p_a in period_averages]
        else:
            period_averages = [p_a / np.mean(p_a, axis=1, keepdims=True)
                               for p_a in period_averages]

        nobs = smoother.data.shape[1]
        seasonal = [np.tile(p_a, (1, nobs // periods[i] + 1))[:, :nobs]
                    for i, p_a in enumerate(period_averages)]

        data = smoother.data
        smooth = smoother.smooth_data
        for season in seasonal:
            if self.method == 'additive':
                smooth += season
            else:
                smooth *= season

        self._store_results(smooth_data=smooth, data=data)

        return self


class KalmanSmoother(_BaseSmoother):
    """KalmanSmoother smoothes the timeseries using the Kalman smoothing
    technique. The Kalman smoother provided here can be represented
    in the state space form. For this reason, it's necessary to provide
    an adequate matrix representation of all the components. It's possible
    to define a Kalman smoother that takes into account the following
    structure present in our series: 'level', 'trend', 'seasonality' and
    'long seasonality'. All these features have an addictive behaviour.

    The KalmanSmoother automatically vectorizes, in an efficient way,
    the desired smoothing operation on all the series received.

    Parameters
    ----------
    component : str
        Specify the patterns and the dinamycs present in our series.
        The possibilities are: 'level', 'level_trend',
        'level_season', 'level_trend_season', 'level_longseason',
        'level_trend_longseason', 'level_season_longseason',
        'level_trend_season_longseason'. Each single component is
        delimited by the '_' notation.

    component_noise : dict
        Specify in a dictionary the noise (in float term) of each single
        component provided in the 'component' argument. If a noise of a
        component, not provided in the 'component' argument, is provided,
        it's automatically ignored.

    observation_noise : float, default=1.0
        The noise level generated by the data measurement.

    n_seasons : int, default=None
        The period of the seasonal component. If a seasonal component
        is not provided in the 'component' argument, it's automatically
        ignored.

    n_longseasons : int, default=None
        The period of the long seasonal component. If a long seasonal
        component is not provided in the 'component' argument, it's
        automatically ignored.

    copy : bool, default=True
        If True, the raw data received by the smoother and the smoothed
        results can be accessed using 'data' and 'smooth_data' attributes.
        This is useful to calculate the intervals. If set to False the
        interval calculation is disabled. In order to save memory, set it to
        False if you are interested only in the smoothed results.

    Attributes
    ----------
    smooth_data : array of shape (series, timesteps)
        Smoothed data derived from the smoothing operation. It is accessible
        after computhing smoothing, otherwise None is returned.

    data : array of shape (series, timesteps)
        Raw data received by the smoother. It is accessible with 'copy'=True
        and  after computhing smoothing, otherwise None is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *
    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=10, timesteps=200,
    ...                       process_noise=10, measure_noise=30)
    >>> smoother = KalmanSmoother(component='level_trend',
    ...                           component_noise={'level':0.1, 'trend':0.1})
    >>> smoother.smooth(data)
    >>> low, up = smoother.get_intervals('kalman_interval')
    """

    def __init__(self, component, component_noise, observation_noise=1.,
                 n_seasons=None, n_longseasons=None, copy=True):
        self.component = component
        self.component_noise = component_noise
        self.observation_noise = observation_noise
        self.n_seasons = n_seasons
        self.n_longseasons = n_longseasons
        self.copy = copy

    def smooth(self, data):
        """Smooth timeseries.

        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,)
               for single timeseries
            Timeseries to smooth. The data are assumed to be in increasing
            time order in each timeseries.

        Returns
        -------
        self : returns an instance of self
        """

        components = ['level', 'level_trend',
                      'level_season', 'level_trend_season',
                      'level_longseason', 'level_trend_longseason',
                      'level_season_longseason', 'level_trend_season_longseason']

        if self.component not in components:
            raise ValueError(
                "'{}' is unsupported. Pass one of {}".format(
                    self.component, components))

        _noise = _check_noise_dict(self.component_noise, self.component)
        data = _check_data_nan(data)

        if self.component == 'level':

            A = [[1]]  # level
            Q = [[_noise['level']]]
            H = [[1]]

        elif self.component == 'level_trend':

            A = [[1, 1],  # level
                 [0, 1]]  # trend
            Q = np.diag([_noise['level'], _noise['trend']])
            H = [[1, 0]]

        elif self.component == 'level_season':

            if self.n_seasons is None:
                raise ValueError(
                    "you should specify n_seasons when using a seasonal component")

            A = np.zeros((self.n_seasons, self.n_seasons))
            A[0, 0] = 1  # level
            A[1, 1:] = [-1.0] * (self.n_seasons - 1)  # season
            A[2:, 1:-1] = np.eye(self.n_seasons - 2)  # season
            Q = np.diag([_noise['level'],
                         _noise['season']] + [0] * (self.n_seasons - 2))
            H = [[1, 1] + [0] * (self.n_seasons - 2)]

        elif self.component == 'level_trend_season':

            if self.n_seasons is None:
                raise ValueError(
                    "you should specify n_seasons when using a seasonal component")

            A = np.zeros((self.n_seasons + 1, self.n_seasons + 1))
            A[:2, :2] = [[1, 1],  # level
                         [0, 1]]  # trend
            A[2, 2:] = [-1.0] * (self.n_seasons - 1)  # season
            A[3:, 2:-1] = np.eye(self.n_seasons - 2)  # season
            Q = np.diag([_noise['level'], _noise['trend'],
                         _noise['season']] + [0] * (self.n_seasons - 2))
            H = [[1, 0, 1] + [0] * (self.n_seasons - 2)]

        elif self.component == 'level_longseason':

            if self.n_longseasons is None:
                raise ValueError(
                    "you should specify n_longseasons when using a "
                    "long seasonal component")

            period_cycle_sin = np.sin(2 * np.pi / self.n_longseasons)
            period_cycle_cos = np.cos(2 * np.pi / self.n_longseasons)

            A = [[1, 0, 0],  # level
                 [0, period_cycle_cos, period_cycle_sin],  # long season
                 [0, -period_cycle_sin, period_cycle_cos]]  # long season
            Q = np.diag([_noise['level'],
                         _noise['longseason'], _noise['longseason']])
            H = [[1, 1, 0]]

        elif self.component == 'level_trend_longseason':

            if self.n_longseasons is None:
                raise ValueError(
                    "you should specify n_longseasons when using a "
                    "long seasonal component")

            period_cycle_sin = np.sin(2 * np.pi / self.n_longseasons)
            period_cycle_cos = np.cos(2 * np.pi / self.n_longseasons)

            A = [[1, 1, 0, 0],  # level
                 [0, 1, 0, 0],  # trend
                 [0, 0, period_cycle_cos, period_cycle_sin],  # long season
                 [0, 0, -period_cycle_sin, period_cycle_cos]]  # long season
            Q = np.diag([_noise['level'], _noise['trend'],
                         _noise['longseason'], _noise['longseason']]),
            H = [[1, 0, 1, 0]]

        elif self.component == 'level_season_longseason':

            if self.n_seasons is None:
                raise ValueError(
                    "you should specify n_seasons when using a seasonal component")

            if self.n_longseasons is None:
                raise ValueError(
                    "you should specify n_longseasons when using a "
                    "long seasonal component")

            period_cycle_sin = np.sin(2 * np.pi / self.n_longseasons)
            period_cycle_cos = np.cos(2 * np.pi / self.n_longseasons)

            A = np.zeros((self.n_seasons + 2, self.n_seasons + 2))
            A[0, 0] = 1  # level
            A[1:3, 1:3] = [[period_cycle_cos, period_cycle_sin],  # long season
                           [-period_cycle_sin, period_cycle_cos]]  # long season
            A[3, 3:] = [-1.0] * (self.n_seasons - 1)  # season
            A[4:, 3:-1] = np.eye(self.n_seasons - 2)  # season
            Q = np.diag([_noise['level'],
                         _noise['longseason'], _noise['longseason'],
                         _noise['season']] + [0] * (self.n_seasons - 2))
            H = [[1, 1, 0, 1] + [0] * (self.n_seasons - 2)]

        elif self.component == 'level_trend_season_longseason':

            if self.n_seasons is None:
                raise ValueError(
                    "you should specify n_seasons when using a seasonal component")

            if self.n_longseasons is None:
                raise ValueError(
                    "you should specify n_longseasons when using a "
                    "long seasonal component")

            period_cycle_sin = np.sin(2 * np.pi / self.n_longseasons)
            period_cycle_cos = np.cos(2 * np.pi / self.n_longseasons)

            A = np.zeros((self.n_seasons + 2 + 1, self.n_seasons + 2 + 1))
            A[:2, :2] = [[1, 1],  # level
                         [0, 1]]  # trend
            A[2:4, 2:4] = [[period_cycle_cos, period_cycle_sin],  # long season
                           [-period_cycle_sin, period_cycle_cos]]  # long season
            A[4, 4:] = [-1.0] * (self.n_seasons - 1)  # season
            A[5:, 4:-1] = np.eye(self.n_seasons - 2)  # season
            Q = np.diag([_noise['level'], _noise['trend'],
                         _noise['longseason'], _noise['longseason'],
                         _noise['season']] + [0] * (self.n_seasons - 2))
            H = [[1, 0, 1, 0, 1] + [0] * (self.n_seasons - 2)]

        kf = simdkalman.KalmanFilter(
            state_transition=A,
            process_noise=Q,
            observation_model=H,
            observation_noise=self.observation_noise)

        smoothed = kf.smooth(data)
        smoothed_obs = smoothed.observations.mean
        cov = np.sqrt(smoothed.observations.cov)

        smoothed_obs = _check_output(smoothed_obs, transpose=False)
        cov = _check_output(cov, transpose=False)
        data = _check_output(data, transpose=False)

        self._store_results(smooth_data=smoothed_obs, cov=cov, data=data)

        return self


class WindowWrapper(_BaseSmoother):
    """WindowWrapper smooths timeseries partitioning them into equal
    sliding segments and treating them as new standalone timeseries.
    The WindowWrapper handles single timeseries. After the sliding windows
    are generated, the WindowWrapper smooths them using the smoother it
    receives as input parameter. In this way, the smoothing can be carried
    out like a multiple smoothing task.

    The WindowWrapper automatically vectorizes, in an efficient way,
    the sliding window creation and the desired smoothing operation.

    Parameters
    ----------
    Smoother : class from tsmoothie.smoother
        Every smoother available in tsmoothie.smoother.
        It computes the smoothing on the series received.

    window_shape : int
        Grather than 1. The shape of the sliding windows used to divide
        the series to smooth.

    step : int, default=1
        The step used to generate the sliding windows.

    Attributes
    ----------
    Smoother : class from tsmoothie.smoother
        Every smoother available in tsmoothie.smoother that was passed to
        WindowWrapper.
        It as the same properties and attributes of every Smoother.

    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *
    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=1, timesteps=200,
    ...                       process_noise=10, measure_noise=30)
    >>> smoother = WindowWrapper(
    ...     LowessSmoother(smooth_fraction=0.3, iterations=1),
    ...     window_shape=30)
    >>> smoother.smooth(data)
    >>> low, up = smoother.get_intervals('prediction_interval')
    """

    def __init__(self, Smoother, window_shape, step=1):
        self.Smoother = Smoother
        self.window_shape = window_shape
        self.step = step

    def smooth(self, data):
        """Smooth timeseries.

        Parameters
        ----------
        data : array-like of shape (1, timesteps) or also (timesteps,)
            Single timeseries to smooth. The data are assumed to be in
            increasing time order.

        Returns
        -------
        self : returns an instance of self
        """

        if not 'tsmoothie.smoother' in str(self.Smoother.__repr__):
            raise ValueError("Use a Smoother from tsmoothie.smoother")

        data = np.asarray(data)
        if np.prod(data.shape) == np.max(data.shape):
            data = data.ravel()[:, None]
        else:
            raise ValueError(
                "The format of data received is not appropriate. "
                "WindowWrapper accepts only univariate timeseries")

        if data.shape[0] < self.window_shape:
            raise ValueError("window_shape must be <= than timesteps")

        data = create_windows(data, window_shape=self.window_shape, step=self.step)
        data = np.squeeze(data, -1)

        self.Smoother.smooth(data)

        return self