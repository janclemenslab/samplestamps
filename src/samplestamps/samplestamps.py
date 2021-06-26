"""Tools for converting between samples and time stamps.

glossary:
    densely stamped - each sample is consecutively time stamped
    sparsely stamped - only selected samples are time stamped - need to provide sample numbers
"""
from .utils import monotonize, interpolator, ismonotonous


class SampStamp():
    """Converts between frames and samples."""

    def __init__(self, sample_times, frame_times=None, sample_numbers=None, frame_numbers=None, frame_samples=None,
                 sample_times_offset=0, frame_times_offset=0,
                 auto_monotonize=True):
        """Get converter.

        Args:
            sample_times(np.ndarray)
            frame_times(np.ndarray)
            sample_number(np.ndarray)
            frame_number(np.ndarray)
            frame_samples
            sample_times_offset(float)
            frame_times_offset(float)
            auto_monotonize(bool)
        """
        # we want:
        # samples -> frames
        # samples -> times
        # frames -> samples
        # frames -> times

        # we need:
        # (samples, times) from DAQ, (frames, times) from video timestamps
        # (samples, times) from DAQ, (samples, frames) from movieframes
        # (samples, frames) from movieframes, (frames, times) from video timestamps (ignore edge case)

        # generate dense x_number arrays
        if sample_numbers is None and sample_times is not None:
            sample_numbers = range(sample_times.shape[0])
        if frame_numbers is None and frame_times is not None:
            frame_numbers = range(frame_times.shape[0])

        # correct for offsets
        sample_times += sample_times_offset
        if frame_times is not None:
            frame_times += frame_times_offset

        if auto_monotonize:
            if sample_times is not None:
                sample_times = monotonize(sample_times)
                sample_numbers = sample_numbers[:sample_times.shape[0]]
            if frame_times is not None:
                frame_times = monotonize(frame_times)
                frame_numbers = frame_numbers[:frame_times.shape[0]]
            if frame_samples is not None:
                frame_samples = monotonize(frame_samples)
                # frame_numbers = frame_numbers[:frame_times.shape[0]]

        # get all interpolators for re-use
        if sample_numbers is not None and sample_times is not None:
            self.samples2times = interpolator(sample_numbers, sample_times)
        if frame_times is None and frame_samples is not None:
            frame_times = self.samples2times(frame_samples)
            # self.frame_times = self.samples2times(frame_samples)

        if frame_numbers is not None and frame_times is not None:
            self.frames2times = interpolator(frame_numbers, frame_times)
        if frame_times is not None and sample_numbers is not None:
            self.times2samples = interpolator(sample_times, sample_numbers)
        if frame_times is not None and frame_numbers is not None:
            self.times2frames = interpolator(frame_times, frame_numbers)

        if frame_samples is not None:
            self.samples2frames = interpolator(frame_samples, frame_numbers, fill_value='extrapolate')

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
    import numpy as np
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
