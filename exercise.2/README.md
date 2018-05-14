# Build the Docker image
```
$ docker build -t ros-dlfad:lunar-perception .
```
# Create out directories
```
$ mkdir -p out/{center,left,right}
```
# Run script
With the data sitting in directory `./data` and output directories `./out/{center,left,right}`:
```
$ docker run -v $(pwd):/srv --rm ros-dlfad:lunar-perception
```
## Specifying input and output directories
```
$ docker run -v $(pwd):/srv --rm ros-dlfad:lunar-perception python exercise.2.reader.py -i 'input.dir' -o 'output.dir'
```
