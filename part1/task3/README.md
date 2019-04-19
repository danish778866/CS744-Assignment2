# Task 3

## Running
```
> cd part1/task3
> ./run.sh -h # Get help text
```

## Examples
```
> ./run.sh -m cluster2 -b 32 -t sync_sgd.py
> ./run.sh -m cluster2 -b 64 -t sync_sgd.py
> ./run.sh -m cluster2 -b 128 -t sync_sgd.py
> ./run.sh -m cluster2 -b 256 -t sync_sgd.py
> ./run.sh -m cluster2 -b 32 -t async_sgd.py
> ./run.sh -m cluster2 -b 64 -t async_sgd.py
> ./run.sh -m cluster2 -b 128 -t async_sgd.py
> ./run.sh -m cluster2 -b 256 -t async_sgd.py
```

## Performance analysis
1. Training logs will get generated in the logs directory.
2. [sar](https://en.wikipedia.org/wiki/Sar_(Unix)) can be used for analyzing
   system metrics.
    * Memory  : `sar -r`
    * CPU     : `sar -u`
    * Network : `sar -n DEV`

**Note:** Keep an eye on the logs generated in logs directory for any errors.
