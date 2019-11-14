# mystyle

A set of functions to tweak matplotlib default plotting style and some useful plots

Author: Juvid Aryaman

## Installation

```
$ git clone https://github.com/jaryaman/mystyle
$ cd mystyle/
$ python setup.py install
```

To set up fonts,
```
$ sudo apt-get install msttcorefonts -qq
$ rm ~/.cache/matplotlib -fr
```

Alternatively, after cloning the repo, if you are on Anaconda use the command
```
$ ln -s mystyle /home/[user_name]/anaconda3/lib/python3.6/site-packages
```
or some appropriate alternative. For example, on Windows,
```
mklink /D C:\Users\[user_name]\Anaconda3\Lib\site-packages C:\Users\[user_name]\Directory\To\mystyle
```

## Usage

For example,

```python
import mystyle.sty as sty
sty.reset_plots() # allow LaTeX, make figure text large, and use an Arial font
```
