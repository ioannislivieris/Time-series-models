'''
A collection of utility functions.
'''

import numpy as np
import scipy.stats as stats
from typing import Iterable


def sim_seasonal_data(n_series, timesteps, measure_noise,
                      freq=None, level=None, amp=None):
    """Generate sinusoidal data with periodic patterns.

    Parameters
    ----------
    n_series : int
        Number of timeseries to generate.

    timesteps : int
        How many timesteps every generated series must have.

    measure_noise : int
        The noise present in the signals.

    freq : int or 1D array-like, default=None
        The frequencies of the sinusoidal timeseries to generate.
        If a single integer is passed, all the series generated have
        the same frequencies. If a 1D array-like is passed, the
        frequencies of timeseries are random sampled from the iterable
        passed. If None, the frequencies are random generated.

    level : int or 1D array-like, default=None
        The levels of the sinusoidal timeseries to generate.
        If a single integer is passed, all the series generated have
        the same levels. If a 1D array-like is passed, the levels
        of timeseries are random sampled from the iterable passed.
        If None, the levels are random generated.

    amp : int or 1D array-like, default=None
        The amplitudes of the sinusoidal timeseries to generate.
        If a single integer is passed, all the series generated have
        the same amplitudes. If a 1D array-like is passed, the amplitudes
        of timeseries are random sampled from the iterable passed.
        If None, the amplitudes are random generated.

    Returns
    -------
    data : array of shape (series, timesteps)
        The generated sinusoidal timeseries.
    """

    if freq is None:
        freq = np.random.randint(3, int(np.sqrt(timesteps)), (n_series, 1))
    elif isinstance(freq, Iterable):
        freq = np.random.choice(freq, size=n_series)[:, None]
    else:
        freq = np.asarray([[freq]] * n_series)

    if level is None:
        level = np.random.uniform(-100, 100, (n_series, 1))
    elif isinstance(level, Iterable):
        level = np.random.choice(level, size=n_series)[:, None]
    else:
        level = np.asarray([[level]] * n_series)

    if amp is None:
        amp = np.random.uniform(3, 100, (n_series, 1))
    elif isinstance(amp, Iterable):
        amp = np.random.choice(amp, size=n_series)[:, None]
    else:
        amp = np.asarray([[amp]] * n_series)

    t = np.repeat([np.arange(timesteps)], n_series, axis=0)
    e = np.random.normal(0, measure_noise, (n_series, timesteps))
    data = level + amp * np.sin(t * (2 * np.pi / freq)) + e

    return data


def sim_randomwalk(n_series, timesteps, process_noise, measure_noise,
                   level=None):
    """Generate randomwalks.

    Parameters
    ----------
    n_series : int
        Number of randomwalks to generate.

    timesteps : int
        How many timesteps every generated randomwalks must have.

    process_noise : int
        The noise present in randomwalks creation.

    measure_noise : int
        The noise present in the signals.

    level : int or 1D array-like, default=None
        The levels of the randomwalks to generate.
        If a single integer is passed, all the randomwalks have
        the same levels. If a 1D array-like is passed, the levels
        of the randomwalks are random sampled from the iterable
        passed. If None, the levels are set to 0 for all the series.

    Returns
    -------
    data : array of shape (series, timesteps)
        The generated randomwalks.
    """

    if level is None:
        level = 0
    if isinstance(level, Iterable):
        level = np.random.choice(level, size=n_series)[:, None]
    else:
        level = np.asarray([[level]] * n_series)

    data = np.random.normal(0, process_noise, size=(n_series, timesteps))
    e = np.random.normal(0, measure_noise, size=(n_series, timesteps))
    data = level + np.cumsum(data, axis=1) + e

    return data


def create_windows(data, window_shape, step=1,
                   start_id=None, end_id=None):
    """Create sliding windows of the same length from the series
    received as input.

    create_windows vectorizes, in an efficient way, the windows creation
    on all the series received.

    Parameters
    ----------
    data : 2D array of shape (timestemps, series)
        Timeseries to slide into equal size windows.

    window_shape : int
        Grather than 1. The shape of the sliding windows used to divide
        the input series.

    step : int, default=1
        The step used to generate the sliding windows. The overlapping
        portion of two adjacent windows can be defined as
        (window_shape - step).

    start_id : int, default=None
        The starting position from where operate slicing. The same for
        all the series. If None, the windows are generated from the index 0.

    end_id : int, default=None
        The ending position of the slicing operation. The same for all the
        series. If None, the windows end on the last position available.

    Returns
    -------
    window_data : 3D array of shape (window_slices, window_shape, series)
        The input data sliced into windows of the same lengths.
    """

    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError(
            "Pass a 2D array-like in the format (timestemps, series)")

    if window_shape < 1:
        raise ValueError("window_shape must be >= 1")

    if start_id is None:
        start_id = 0

    if end_id is None:
        end_id = data.shape[0]

    data = data[int(start_id):int(end_id), :]

    window_shape = (int(window_shape), data.shape[-1])
    step = (int(step),) * data.ndim

    slices = tuple(slice(None, None, st) for st in step)
    indexing_strides = data[slices].strides

    win_indices_shape = ((np.array(data.shape) - window_shape) // step) + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(data.strides))

    window_data = np.lib.stride_tricks.as_strided(
        data, shape=new_shape, strides=strides)

    return np.squeeze(window_data, 1)


def sigma_interval(true, prediction, n_sigma):
    """Compute smoothing intervals as n_sigma times the residuals of the
    smoothing process.

    Returns
    -------
    low : array
        Lower bands.

    up : array
        Upper bands.
    """

    std = np.nanstd(true - prediction, axis=1, keepdims=True)

    low = prediction - n_sigma * std
    up = prediction + n_sigma * std

    return low, up


def kalman_interval(true, prediction, cov, confidence=0.05):
    """Compute smoothing intervals from a Kalman smoothing process.

    Returns
    -------
    low : array
        Lower bands.

    up : array
        Upper bands.
    """

    g = stats.norm.ppf(1 - confidence / 2)

    resid = true - prediction
    std_err = np.sqrt(np.nanmean(np.square(resid), axis=1, keepdims=True))

    low = prediction - g * (std_err * cov)
    up = prediction + g * (std_err * cov)

    return low, up


def confidence_interval(true, prediction, exog, confidence,
                        add_intercept=True):
    """Compute confidence intervals for regression tasks.

    Returns
    -------
    low : array
        Lower bands.

    up : array
        Upper bands.
    """

    if exog.ndim == 1:
        exog = exog[:, None]

    if add_intercept:
        exog = np.concatenate([np.ones((len(exog), 1)), exog], axis=1)

    N = exog.shape[0]
    d_free = exog.shape[1]
    t = stats.t.ppf(1 - confidence / 2, N - d_free)

    resid = true - prediction
    mse = (np.square(resid).sum(axis=1, keepdims=True) / (N - d_free)).T

    hat_matrix_diag = (exog * np.linalg.pinv(exog).T).sum(axis=1, keepdims=True)
    predict_mean_se = np.sqrt(hat_matrix_diag * mse).T

    low = prediction - t * predict_mean_se
    up = prediction + t * predict_mean_se

    return low, up


def prediction_interval(true, prediction, exog, confidence,
                        add_intercept=True):
    """Compute prediction intervals for regression tasks.

    Returns
    -------
    low : array
        Lower bands.

    up : array
        Upper bands.
    """

    if exog.ndim == 1:
        exog = exog[:, None]

    if add_intercept:
        exog = np.concatenate([np.ones((len(exog), 1)), exog], axis=1)

    N = exog.shape[0]
    d_free = exog.shape[1]
    t = stats.t.ppf(1 - confidence / 2, N - d_free)

    resid = true - prediction
    mse = (np.square(resid).sum(axis=1, keepdims=True) / (N - d_free)).T

    covb = np.linalg.pinv(np.dot(exog.T, exog))[..., None] * mse
    predvar = mse + (exog[..., None] *
                     np.dot(covb.transpose(2, 0, 1), exog.T).T).sum(1)
    predstd = np.sqrt(predvar).T

    low = prediction - t * predstd
    up = prediction + t * predstd

    return low, up


def _check_noise_dict(noise_dict, component):
    """Ensure noise compatibility for the noises of the components
    provided when building a state space model.

    Returns
    -------
    noise_dict : dict
        Checked input.
    """

    sub_component = component.split('_')

    if isinstance(noise_dict, dict):
        for c in sub_component:

            if c not in noise_dict:
                raise ValueError(
                    "You need to provide noise for '{}' component".format(c))

            if noise_dict[c] < 0:
                raise ValueError(
                    "noise for '{}' must be >= 0".format(c))

        return noise_dict

    else:
        raise ValueError(
            "noise should be a dict. Received {}".format(type(noise_dict)))


def _check_knots(knots, min_n_knots):
    """Ensure knots compatibility for the knots provided when building
    bases for linear regression.

    Returns
    -------
    knots : array
        Checked input.
    """

    knots = np.asarray(knots, dtype=np.float64)

    if np.prod(knots.shape) == np.max(knots.shape):
        knots = knots.ravel()

    if knots.ndim != 1:
        raise ValueError("knots must be a list or 1D array")

    knots = np.unique(knots)
    min_k, max_k = knots[0], knots[-1]

    if min_k < 0 or max_k > 1:
        raise ValueError("Every knot must be in the range [0,1]")

    if min_k > 0:
        knots = np.append(0., knots)

    if max_k < 1:
        knots = np.append(knots, 1.)

    if knots.shape[0] < min_n_knots + 2:
        raise ValueError(
            "Provide at least {} knots in the range (0,1)".format(min_n_knots))

    return knots


def _check_weights(weights, basis_len):
    """Ensure weights compatibility for the weights provided in
    linear regression applications.

    Returns
    -------
    weights : array
        Checked input.
    """

    if weights is None:
        return np.ones(basis_len, dtype=np.float64)

    weights = np.asarray(weights, dtype=np.float64)

    if np.prod(weights.shape) == np.max(weights.shape):
        weights = weights.ravel()

    if weights.ndim != 1:
        raise ValueError("Sample weights must be a list or 1D array")

    if weights.shape[0] != basis_len:
        raise ValueError(
            "Sample weights length must be equal to timesteps "
            "dimension of the data received")

    if np.any(weights < 0):
        raise ValueError("weights must be >= 0")

    if np.logical_or(np.isnan(weights), np.isinf(weights)).any():
        raise ValueError("weights must not contain NaNs or Inf")

    return weights


def _check_data(data):
    """Ensure data compatibility for the series received by the smoother.

    Returns
    -------
    data : array
        Checked input.
    """

    data = np.asarray(data)

    if np.prod(data.shape) == np.max(data.shape):
        data = data.ravel()

    if data.ndim > 2:
        raise ValueError(
            "The format of data received is not appropriate. "
            "Pass an object with data in this format (series, timesteps)")

    if data.ndim == 0:
        raise ValueError(
            "Pass an object with data in this format (series, timesteps)")

    if data.dtype not in [np.float16, np.float32, np.float64,
                          np.int8, np.int16, np.int32, np.int64]:
        raise ValueError("data contains not numeric types")

    if np.logical_or(np.isnan(data), np.isinf(data)).any():
        raise ValueError("data must not contain NaNs or Inf")

    return data.T


def _check_data_nan(data):
    """Ensure data compatibility for the series received by the smoother.
    (Without checking for inf and nans).

    Returns
    -------
    data : array
        Checked input.
    """

    data = np.asarray(data)

    if np.prod(data.shape) == np.max(data.shape):
        data = data.ravel()

    if data.ndim > 2:
        raise ValueError(
            "The format of data received is not appropriate. "
            "Pass an objet with data in this format (series, timesteps)")

    if data.ndim == 0:
        raise ValueError(
            "Pass an object with data in this format (series, timesteps)")

    if data.dtype not in [np.float16, np.float32, np.float64,
                          np.int8, np.int16, np.int32, np.int64]:
        raise ValueError("data contains not numeric types")

    return data


def _check_output(output, transpose=True):
    """Ensure output compatibility for the series returned by the smoother.

    Returns
    -------
    output : array
        Checked input.
    """

    if transpose:
        output = output.T

    if output.ndim == 1:
        output = output[None, :]

    return output


def _id_nb_bootstrap(n_obs, block_length):
    """Create bootstrapped indexes with the none overlapping block bootstrap
    ('nbb') strategy given the number of observations in a timeseries and
    the length of the blocks.

    Returns
    -------
    _id : array
        Bootstrapped indexes.
    """

    n_blocks = int(np.ceil(n_obs / block_length))
    nexts = np.repeat([np.arange(0, block_length)], n_blocks, axis=0)

    blocks = np.random.permutation(
        np.arange(0, n_obs, block_length)
    ).reshape(-1, 1)

    _id = (blocks + nexts).ravel()[:n_obs]

    return _id


def _id_mb_bootstrap(n_obs, block_length):
    """Create bootstrapped indexes with the moving block bootstrap
    ('mbb') strategy given the number of observations in a timeseries
    and the length of the blocks.

    Returns
    -------
    _id : array
        Bootstrapped indexes.
    """

    n_blocks = int(np.ceil(n_obs / block_length))
    nexts = np.repeat([np.arange(0, block_length)], n_blocks, axis=0)

    last_block = n_obs - block_length
    blocks = np.random.randint(0, last_block, (n_blocks, 1))
    _id = (blocks + nexts).ravel()[:n_obs]

    return _id


def _id_cb_bootstrap(n_obs, block_length):
    """Create bootstrapped indexes with the circular block bootstrap
    ('cbb') strategy given the number of observations in a timeseries
    and the length of the blocks.

    Returns
    -------
    _id : array
        Bootstrapped indexes.
    """

    n_blocks = int(np.ceil(n_obs / block_length))
    nexts = np.repeat([np.arange(0, block_length)], n_blocks, axis=0)

    last_block = n_obs
    blocks = np.random.randint(0, last_block, (n_blocks, 1))
    _id = np.mod((blocks + nexts).ravel(), n_obs)[:n_obs]

    return _id


def _id_s_bootstrap(n_obs, block_length):
    """Create bootstrapped indexes with the stationary bootstrap
    ('sb') strategy given the number of observations in a timeseries
    and the length of the blocks.

    Returns
    -------
    _id : array
        Bootstrapped indexes.
    """

    random_block_length = np.random.poisson(block_length, n_obs)
    random_block_length[random_block_length < 3] = 3
    random_block_length[random_block_length >= n_obs] = n_obs
    random_block_length = random_block_length[random_block_length.cumsum() <= n_obs]
    residual_block = n_obs - random_block_length.sum()
    if residual_block > 0:
        random_block_length = np.append(random_block_length, residual_block)

    n_blocks = random_block_length.shape[0]
    nexts = np.zeros((n_blocks, random_block_length.max() + 1))
    nexts[np.arange(n_blocks), random_block_length] = 1
    nexts = np.flip(nexts, 1).cumsum(1).cumsum(1).ravel()
    nexts = (nexts[nexts > 1] - 2).astype(int)

    last_block = n_obs - random_block_length.max()
    blocks = np.zeros(n_obs, dtype=int)
    if last_block > 0:
        blocks = np.random.randint(0, last_block, n_blocks)
        blocks = np.repeat(blocks, random_block_length)
    _id = blocks + nexts

    return _id