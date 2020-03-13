# Samplestamps

Code for converting between time stamps and samples/frames.

## Installation
`pip install samplestamps`

## Usage

```python
from samplestamps import SampStamp
ss = SampStamp(sample_times=daq_stamps[:,0],
               frame_times=cam_stamps[:,0],
               sample_numbers=daq_samplenumber[:,0])
ledonset_sample = 80000.0+np.argmax(daq_samples[80000:,-1]>0.05)
print(f'led onset at sample {ledonset_sample} corresponding to')
print(f'   seconds: {ss.sample_time(ledonset_sample):1.3f}')
print(f'   frame: {ss.frame(ledonset_sample):1.0f}.')
```