# stacking-robot
6.421 final project 

## Development
An installation of manipulation with pip should work (I haven't tried). 
Using the Dockerfile should directly work as well.

Run main.py to launch the simulation.


## Docker cmds

```
sudo docker build -t stacking .
sudo docker run -it -p 7000:7000 -p 8888:8888 stacking /bin/bash
cd workspace
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```