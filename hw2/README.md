# Homework Assignment #2
This homework assignment will get you familiar with two important concepts:
- Differentiable Simulators
- Knowledge Guided Machine Learning

Follow the instructions below to set up your virtual environment.

### Step 1: Create and enter a virtual environment with python3.12:
`python3.12 -m venv myenv`\
`source myenv/bin/activate`

### Step 2: Clone this repository
`git clone https://github.com/AI-Complexity-Lab/cse598.git`

### Step 3: Go to the Neural Operator HW Assignment Folder
`cd cse598-hw/homework\ 2`

### Step 4: Installation Process
The installation process is a bit complex, given that the libraries being installed often have dependency issues. Be sure to follow the instructions carefully.\
\
Make sure pip is installed in your virtual environment.\
`pip install -r requirements.txt`\
\
Next, uninstall phiflow using: `pip uninstall phiflow`\
Now, reinstall the requirements (this is to resolve a dependency issue): `pip install -r requirements.txt`\
\
*If you encounter an issue with ffmpeg, install conda and run this command:*\
`conda install -c conda-forge ffmpeg`
\
*If you encounter mkl warnings, install an older version of mkl running this command:*\
`pip install mkl==2022.2.1`

### Step 5: Complete the assignment
Head to the jupyter notebook interface by entering this command:\
`jupyter notebook`\
Next, simply click into the notebook file *'___.ipynb'* of each assignment and complete them.
