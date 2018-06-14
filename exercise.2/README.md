# Build the Docker image
```
$ docker build -t ros-dlfad:lunar-perception .
```
# Read bag files
With training data bags sitting in directory `./training.data.bags`:
```
$ docker run -v $(pwd):/srv --rm ros-dlfad:lunar-perception
```
Will create and fill `./training.data.files/{left,center,right}` directories
and `{.camera,.steering,training.data}.csv`
## Specifying input and output directories
```
$ docker run -v $(pwd):/srv --rm ros-dlfad:lunar-perception python exercise.2.reader.py -i 'input.dir' -o 'output.dir'
```
# Create LMDB
With training data files sitting in directory `./training.data.files`:
```
$ python exercise.2.create.lmdb.py
```
Specifying input and output directories same as [before](#Specifying-input-and-output-directories)
# Visualize training data
With training data lmdb sitting in directory `./training.data`:
```
$ python exercise.2.visualizer.py
```
Specifying input and output directories same as [before](#Specifying-input-and-output-directories)