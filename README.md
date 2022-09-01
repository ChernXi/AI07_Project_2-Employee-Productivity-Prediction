# AI07_Project_2-Employee-Productivity-Prediction

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

## Model Summary
![image](https://user-images.githubusercontent.com/108325848/184591882-f8cbf66b-e746-4519-854d-a883e284eb96.png)

# Result
## Sewing Model
Performance of the model:<br>
![image](https://user-images.githubusercontent.com/108325848/187818155-987ac567-a591-4c29-bcdb-46713ea8394f.png)<br>
![image](https://user-images.githubusercontent.com/108325848/187818040-9d7649d7-e8fc-4119-b2c3-feca82dcaa2a.png)<br>

![image](https://user-images.githubusercontent.com/108325848/187819200-d1839d0e-8602-4d72-8530-963648cd29c5.png)<br>

The shaded region is surrounded by a 95% confidence interval, within which we are 95% certain that the **mean value** of prediction lies.<br>
This shall not be confused with the dotted line, which is the 95% prediction interval, within which 95% of our prediction data is contained.

## Finishing Model
Performance of the model:<br>
![image](https://user-images.githubusercontent.com/108325848/187713121-4dad4cd0-bc1f-4f25-a8e8-cef52ebfff7a.png)<br>
![image](https://user-images.githubusercontent.com/108325848/187818313-85ecd2c0-c0a9-4c40-ab86-e56663897097.png)<br>

![image](https://user-images.githubusercontent.com/108325848/187712523-078f35c1-7cea-4707-bcdb-a69b0e7a0bd2.png)<br>

## Conclusion
"All model are wrong, but some are useful."~[George Box](https://en.wikipedia.org/wiki/All_models_are_wrong)<br>

The sewing model is reasonably well fit, as evidenced by the 95% confidence interval passing through the origin, and the majority of the prediction data converged to a straight line. It also has a validation mean absolute percentage error of 0.029 and a mean absolute percentage error of about 5%. <br> 

The finishing model is not fitting so well, as the 95% confidence interval does not pass through the origin and the majority of prediction data appears to be less convergent. The linear fit, on the other hand, shows that the prediction increases linearly with the real data. So it stands to reason that the model has the potential to be improved if more data is provided. <br>  







