# mystyle

A set of functions to tweak matplotlib default plotting style and some useful plots

Author: Juvid Aryaman

## Installation

```
$ git clone https://github.com/jaryaman/mystyle
$ cd mystyle/
$ python setup.py install
```

Alternatively, after cloning the repo, if you are on Anaconda use the command
```
$ ln -s mystyle /home/[user_name]/anaconda3/lib/python3.6/site-packages
```
(or some appropriate alternative).

## Usage

For example,

```python
import mystyle.sty as sty
sty.reset_plots() # allow LaTeX, make figure text large, and use an Arial font
```
