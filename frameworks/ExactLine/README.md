# ExactLine

We use the ExactLine implemented by authors of GenProver. The source code can be downloaded [here](https://openreview.net/forum?id=HJxRMlrtPH).

The implmentation of ExactLine and GenProver are almost the same, except that GenProver merges segments in intermediate outputs as box/polyhedra. Thus, to use ExactLine, you only need to set

```python
use_clustr = None
```

in the implementation of GenProver.