"""Tools for converting between samples and time stamps.

glossary:
    densely stamped - each sample is consecutively time stamped
    sparsely stamped - only selected samples are time stamped - need to provide sample numbers
"""
import numpy as np
import operator as op
from datetime import datetime


def ismonotonous(x, direction='increasing' , strict=True):
    """Check if vector is monotonous.

    Args:
        x(np.ndarray)
        direction(str): 'increasing' or 'decreasing'
        strict(bool): defaults to True
    Returns:
        (bool)
    """
    allowed_directions = ['increasing', 'decreasing']
    if direction not in allowed_directions:
        raise ValueError(f'Direction "{direction}" must be in {allowed_directions}.')

    if direction=='decreasing':
        x = -x

    if strict:
        comp_op = op.gt  # >
    else:
        comp_op = op.ge  # >=

    return np.all(comp_op(x[1:],x[:-1]))


def monotonize(x, direction='increasing', strict=True):
    """Cut trailing non-monotonous values.

    Args:
        x
        direction - montonously 'increasing' (default) or 'decreasing'
        strict - strictly (default) or non-strictly monotonous
    Returns:
        truncated array
    """
    allowed_directions = ['increasing', 'decreasing']
    if direction not in allowed_directions:
        raise ValueError(f'Direction "{direction}" must be in {allowed_directions}.')

    if strict:
        comp_op = op.le  # >=
    else:
        comp_op = op.lt  # >

    if direction=='decreasing':
        last_idx = np.argmax(comp_op(x[:-1],x[1:]))+1
    else:
        last_idx = np.argmax(comp_op(x[1:],x[:-1]))+1

    return x[:last_idx]


def interpolator(x, y, fill_value='extrapolate'):
    import scipy.interpolate
    return scipy.interpolate.interp1d(x, y, fill_value=fill_value)


def time_from_log(logfilename, line_number=1):
    """Parse time stamp from a specified lines in a log file.

    Args:
        logfilename(str)
        line_number(int): line in the log file from which parse the time stamp (defaults to 1 - will read the first line (not 0-indexed!))
    Returns:
        (datetime) time stamp
    """
    with open(logfilename,'r') as f:
        for _ in range(line_number):
            current_line = f.readline()

    current_line_parts = current_line.partition(' ')[0]
    return datetime.strptime(current_line_parts, '%Y-%m-%d,%H:%M:%S.%f')


def samplenumber_from_timestamps(target_time, timestamps, sample_at_timestamps=None):
    """Gets samplenumber from timestamps given time.

    Args:
        target_time (numpy.ndarrary): time of desired sample
        timestamps (numpy.ndarrary): list of timestamps
        sample_at_timestamps: can be provided for sparsely stamped data, the sample number for each timestamp
    Returns:
        samplenumber at target time (as an index, starts at 0, can be <0 if target is before first timestamps) (np.intp)
    """
    if not ismonotonous(timestamps, strict=True):
        raise ValueError(f'Timestamps must increase strictly monotonously.')

    if sample_at_timestamps is None:
        sample_at_timestamps = range(timestamps.shape[0])

    f = interpolator(timestamps, sample_at_timestamps)
    samplenumber = np.intp(np.round(f(target_time)))

    return samplenumber


def samplerange_from_timestamps(target_epoch, timestamps, sample_at_timestamps=None):
    """Gets range of samples from timestamps given a epoch defined by start and stop time.

    Args:
        target_epoch (numpy.ndarrary): start and stop time
        timestamps (numpy.ndarrary): list of timestamps
        sample_at_timestamps: can be provided for sparsely stamped data, the sample number for each timestamp
    Returns:
        range of samples spanning epoch (as an indices, starts at 0, can be <0 if targets extends to before first timestamps)
    """
    samplenumber_start = samplenumber_from_timestamps(target_epoch[0], timestamps, sample_at_timestamps)
    samplenumber_end = samplenumber_from_timestamps(target_epoch[1], timestamps, sample_at_timestamps)
    return range(samplenumber_start, samplenumber_end)


def timestamp_from_samplenumber(samplenumber, timestamps, sample_at_timestamps=None):
    """Gets samplenumber from timestamps given time.

    Args:
        samplenumber (numpy.ndarrary): sample number for which we want the time stamp
        timestamps (numpy.ndarrary): list of timestamps
        sample_at_timestamps: can be provided for sparsely stamped data, the sample number for each timestamp
    Returns:
        time stamp for that sample (float)
    """
    if not ismonotonous(timestamps, strict=True):
        raise ValueError(f'Timestamps must increase strictly monotonously.')

    if sample_at_timestamps is None:
        sample_at_timestamps = range(timestamps.shape[0])

    f = interpolator(sample_at_timestamps, timestamps)
    timestamp = f(samplenumber)

    return timestamp


def samples_from_samples(sample_in, in_stamps, in_samplenumber=None, out_stamps=None, out_samplenumber=None):
    """Convert between different sampling grids via a common clock.
    Args:
        sample_in: sample number in INPUT sampling grid
        in_stamps: time stamps of samples numbers `in_samplenumber` in OUTPUT sampling grid
        in_samplenumber: sample numbers for INPUT timestamps (defaults to None for densely stamped data)
        out_stamps: time stamps of samples numbers `out_samplenumber` in OUTPUT sampling grid
        out_samplenumber: sample numbers for INPUT timestamps (defaults to None for densely stamped data)
    Returns:
        sample number in the OUTPUT sampling grid corresponding to sample_in
    """
    time_in = timestamp_from_samplenumber(sample_in, in_stamps, in_samplenumber)
    sample_out = samplenumber_from_timestamps(time_in, out_stamps, out_samplenumber)
    return sample_out


class SampStamp():
    """Converts between frames and samples."""

    def __init__(self, sample_times, frame_times, sample_numbers=None, frame_numbers=None, sample_times_offset=0, frame_times_offset=0, auto_monotonize=True):
        """Get converter.

        Args:
            sample_times(np.ndarray)
            frame_times(np.ndarray)
            sample_number(np.ndarray)
            frame_number(np.ndarray)
            sample_times_offset(float)
            frame_times_offset(float)
            auto_monotonize(bool)
        """

        # generate dense x_number arrays
        if sample_numbers is None:
            sample_numbers = range(sample_times.shape[0])
        if frame_numbers is None:
            frame_numbers = range(frame_times.shape[0])

        # correct for offsets
        sample_times += sample_times_offset
        frame_times += frame_times_offset

        if auto_monotonize:
            sample_times = monotonize(sample_times)
            sample_numbers = sample_numbers[:sample_times.shape[0]]
            frame_times = monotonize(frame_times)
            frame_numbers = frame_numbers[:frame_times.shape[0]]

        # get all interpolators for re-use
        self.samples2times = interpolator(sample_numbers, sample_times)
        self.frames2times = interpolator(frame_numbers, frame_times)
        self.times2samples = interpolator(sample_times, sample_numbers)
        self.times2frames = interpolator(frame_times, frame_numbers)

    def frame(self, sample):
        """Get frame number from sample number."""
        return self.times2frames(self.sample_time(sample))

    def sample(self,  frame):
        """Get sample number from frame number."""
        return self.times2samples(self.frame_time(frame))

    def frame_time(self, frame):
        """Get time of frame number."""
        return self.frames2times(frame)

    def sample_time(self, sample):
        """Get time of sample number."""
        return self.samples2times(sample)


def test():
    # test cases:
    inc_strict = np.array([0, 1, 2, 3])
    inc_nonstrict = np.array([0, 1, 2, 2])
    dec_strict = inc_strict[::-1]
    dec_nonstrict = inc_nonstrict[::-1]
    assert ismonotonous(inc_strict, direction='increasing' , strict=True)==True
    assert ismonotonous(inc_strict, direction='increasing' , strict=False)==True
    assert ismonotonous(inc_nonstrict, direction='increasing' , strict=True)==False
    assert ismonotonous(inc_nonstrict, direction='decreasing' , strict=False)==False
    assert ismonotonous(dec_strict, direction='decreasing' , strict=True)==True
    assert ismonotonous(dec_nonstrict, direction='decreasing' , strict=True)==False
    assert ismonotonous(np.array([1]), direction='increasing' , strict=True)==True
    assert ismonotonous(np.array([1]), direction='increasing' , strict=False)==True

    x = np.array([0, 1, 2, 2, 1])
    print(f"montonize {x}")
    print(f"  strict, inc: {monotonize(x)}")
    assert np.all(monotonize(x)==[0,1,2])
    print(f"  strict, dec: {monotonize(x, direction='decreasing')}")
    assert np.all(monotonize(x, direction='decreasing')==[0])
    print(f"  nonstrict, in: {monotonize(x, strict=False)}")
    assert np.all(monotonize(x, strict=False)==[0,1,2,2])

    x = np.array([2, 1, 0, 0, 1])
    print(f"montonize {x}")
    print(f"  strict, inc: {monotonize(x)}")
    assert np.all(monotonize(x)==[2])
    print(f"  strict, dec: {monotonize(x, direction='decreasing')}")
    assert np.all(monotonize(x, direction='decreasing')==[2,1,0])
    print(f"  nonstrict, dec: {monotonize(x, strict=False, direction='decreasing')}")
    assert np.all(monotonize(x, strict=False, direction='decreasing')==[2,1,0,0])


if __name__=='__main__':
    test()
