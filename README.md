# gumbel-rl-gridworld
The use of Gumbel softmax for single-agent RL in a simple gridworld

## Dependency
Known dependencies are:
```
python (3.6.5)
pip (3.6)
virtualenv
```

## Setup
To avoid any conflict, please install Python virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/):
```
pip3.6 install --upgrade virtualenv
```

## Run
After all the dependencies are installed, please run the code by running the following script.  
The script will start the virtual environment, install remaining Python dependencies in `requirements.txt`, and run the code.  
```
./_train.sh
```

## Result
The code should reproduce the following result (e.g., 3x9 gridworld):
![alt text][result]

[result]: https://github.com/dkkim93/gumbel-rl-gridworld/blob/master/result.png "Result on 3x9"
