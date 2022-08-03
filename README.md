# Project_2-Employee-Productivity-Prediction

This project aims to show how to construct a neural network model to predict the Employee Productivity in European garment industry.
The data is collected from European country at the year of 1978.
You can get the data from [here](https://archive.ics.uci.edu/ml/datasets/Productivity+Prediction+of+Garment+Employees).

## Summary of the Dataset
There are 15 attributes in this dataset. We will make use of the first 14 attributes as the features, and use them to launch a multilinear regression traning by using a dense neural network. Our goal is to train out a model that can predict the actual productivity, which is the last attribute of the dataset, with our targeted mean absolute error percentage less than 10%.

The 15 attributes in dataset are summarized as below.

01 date : Date in MM-DD-YYYY <br>
02 day : Day of the Week <br>
03 quarter : A portion of the month. A month was divided into four quarters <br>
04 department : Associated department with the instance <br>
05 team_no : Associated team number with the instance <br>
06 no_of_workers : Number of workers in each team <br>
07 no_of_style_change : Number of changes in the style of a particular product <br>
08 targeted_productivity : Targeted productivity set by the Authority for each team for each day <br>
09 smv : Standard Minute Value, it is the allocated time for a task <br>
10 wip : Work in progress. Includes the number of unfinished items for products <br>
11 over_time : Represents the amount of overtime by each team in minutes <br>
12 incentive : Represents the amount of financial incentive (in BDT) that enables or motivates a particular course of action <br>
13 idle_time : The amount of time when the production was interrupted due to several reasons <br>
14 idle_men : The number of workers who were idle due to production interruption <br>
15 actual_productivity : The actual % of productivity that was delivered by the workers. It ranges from 0-1. <br>




