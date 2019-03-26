from setuptools import setup

setup(
	name='mystyle',
	version='0.0.1',
	description='A set of functions to tweak matplotlib default plotting style and some common useful plots',
	licence='MIT',
	url='https://github.com/jaryaman/mystyle/',
	author='Juvid Aryaman',
	author_email='j.aryaman25@gmail.com',
	keywords=['matplotlib'],
	packages=['mystyle'],
	install_requires=['numpy','scipy','matplotlib','pandas','warnings','sklearn','IPython'],
)
