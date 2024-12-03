# Homework Assignment #3
This homework assignment will get you familiar with two important concepts:
- Discrepancy Modeling for Epidemiology
- Symbolic Regression

Follow the instructions below to set up your virtual environment.

**Note, please use conda instead of a python venv. Instructions below:**

### Step 1: Clone this repository
`git clone https://github.com/AI-Complexity-Lab/cse598.git`

### Step 2: Head to homework 3
`cd cse598/hw3`

### Step 3a: Create a virtual environment with conda with the correct dependencies:
`conda env create -f environment.yml`

### Step 3b: Activate the environment
`conda activate cse598`

### Step 4a: Alternatively, create a virtual environment with pip using python3.12:
`python3.12 -m venv myenv`

### Step 4b: Activate this environment
`source myenv/bin/activate`

### Step 4c: Install the dependencies
`pip install -r requirements.txt`

### Step 5: Complete both assignments
Head to the 2 jupyter notebooks either through the jupyter interface by entering this command:\
`jupyter notebook`

Or by using vscode.

Then, complete both assignments. Happy coding!

### Step 6: Additional setup for Symbolic Regression Assignment
Please create a directory **inside the hw3 directory** called `weights`.
Next, please download both the 10M and 100M datasets (as specified using the links in the notebook) and place them in the `weights` folder - they were not provided in this repo because the datasets are quite large (300MB each).
