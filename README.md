# Milk Quality Classification

This project was made for Machine Learning to practice and compare classification between random forest and support vector. It includes two files: _milknew.csv_ and _RandomForest.py_.

**milknew.csv**: the dataset, taken from a public [milk quality dataset](https://www.kaggle.com/datasets/yrohit199/milk-quality). Important columns include pH, turbidity, temperature, and fat, among others.

**RandomForest.py**: the classification program.

First it runs a Random Forest with the optimal parameters between 3-5 depth, 2/5/10 min_sample_split and 1/5/10 min_sample_leaf. It displays the parameters that were most important, as represented by the y-axis (contribution to accuracy):
![Figure_1](https://github.com/user-attachments/assets/4f9dd1d2-98ea-4d59-b055-fdd4acc7f5f3)
_Notice pH and temperature make up most of the score, as expected._

...and then outputs a confusion matrix for the results:
![image](https://github.com/user-attachments/assets/dd00bc02-1170-4bba-95d5-a2764e591f8e)
_The biggest discrepency was High being confused with medium quality._

Next, it runs a similar process for SVC. Due to SVC finding a margin between two classes.
![image](https://github.com/user-attachments/assets/68fb2474-677b-4139-949a-8e900767787e)
