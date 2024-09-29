### Overview
- The pipeline is modified from [PyICP-SLAM](https://github.com/gisbi-kim/PyICP-SLAM) and composed of three parts
    1. Odometry: [GeoTransformer](https://github.com/qinzheng93/GeoTransformer/) and [FastMAC](https://github.com/Forrest-110/FastMAC)
        - In here, Point-to-point and frame-to-frame (i.e., no local mapping)
    2. Loop detection: [Scan Context (IROS 18)](https://github.com/irapkaist/scancontext)
       -  Reverse loop detection is supported.
    3. Back-end (graph optimizer): [miniSAM](https://github.com/dongjing3309/minisam)
       - Python API

- This is a simple python usage example without any parameter tuning or efficiency optimization. So the results may not be good enough.

### How to use
Just run

```sh
$ python3 pipeline.py
```

The details of parameters are eaily found in the argparser in that .py file.
