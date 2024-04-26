
# EE 658/758 - Assignment 5 
### by Rohith Nagineni

This Web application leverages Python and Streamlit which allows users to have UI where they can select the dataset and ML model and also user can give input feature values to get predictions.
## Requirments
- Python3
- Streamlit
- Sklearn
- Matplotlib
- NumPy

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install Streamlit
pip install Scikit-learn
pip install Matplotlib
pip install  NumPy
```
### Running the Application
- Open a VScode and open the folder where your N1.py is located.
- Open a terminal.
- Run the Streamlit application

```bash
streamlit run main.py
```
After running this comand the application will open in your browser
## How to access the Front-end UI
- Select the dataset (IRIS or Digits) from the sidebar.
- Choose the machine learning classifier from the available options.
- Input the feature values through the sidebar.
- The prediction result will be displayed in the main panel.
- Accuracy comparison graph and confusion matrix will be shown in the main panel. 