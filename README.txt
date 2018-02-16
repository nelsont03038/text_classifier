This document will describe how to install and run the simple text classifier web application.  The app is implemented in Python and uses the Django web application framework.  It is simple to install the requirements and run a development web server that will host the application. 



Get the project by either saving and decompressing the attached archive, or clone it from GitHub.

Assuming you have git installed, open a terminal window and type:

git clone git://github.com/nelsont03038/text_classifier

If you do not have git, simply download the attached archive and decompress it.



Requirements

	You will need a functioning Python 3.x environment.
	The following Python packages are required: numpy, pandas, sklearn, django.
		
Use pip to install packages.  pip is already installed if you are using a modern version of Python.  If you need to manually install pip, go here for instructions: https://pip.pypa.io/en/stable/installing/

You might want to consider a separate virtual python environment to isolate these packages if you do not want to have them forever.  If you do that, then activate the virtual environment before you install the packages.
	
The quick and easy way to install the requirements is to simply open a terminal and type:
	pip install -r /path/to/requirements.txt
Of course change the path to the location of the requirements.txt file.
	
Or if you need/want more control over your packages, install them individually with pip:
	
To install numpy do: 
	pip install numpy 
To install pandas do: 
	pip install pandas
To install sklearn do: 
	pip install sklearn
To install Django do:
	pip install Django==1.11

	
Once you have python and the needed packages installed, start the development web server to run on localhost port 8000.  To do this, open up a terminal and change directories to the one containing the Django project (the "stc_project" directory).  You should see a file "manage.py".  

Type at the command line: 
	python manage.py runserver

This will start up a development web server running the web application.  

To access the web application, point your web browser to: http://127.0.0.1:8000/ (or http://localhost:8000/) 

To shutdown the server when you are done, type <ctrl>-<break> in Windows, <ctrl>-c in MacOS and Linux.

