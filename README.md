# Productivity Prediction of Garment Employees Data Set 

## 1. Motivation
The garment industry is vital to human civilization and has a massive global demand.<br>

To estimate their garment production capacity and ensure that production is completed on time, large garment production companies must track, analyse, and predict the productivity of their working teams. This is critical in order to maintain the company's reputation and avoid paying breach of contract penalties.<br>

Furthermore, a good prediction model was critical in the future development of the garment company because it provided information to streamline production lines and optimise production efficiency.<br>

## 2. Objective
(i) Construct a good prediction model that can be used to predict the employee productivity in European garment Industry.<br>

(ii) Data mining to determine which attributes in the European garment industry are highly correlated with employee productivity and what their optimal value is.<br>

(iii) As a demonstration to show how to construct a dense neural network.<br>

(iv) As a demonstration to show how to do data cleaning, data preparation, data analysis and data exploration.<br>

## 3. The Garment Employee Dataset
### 3.1. Download link
You can get the dataset download link from [here](https://archive.ics.uci.edu/ml/datasets/Productivity+Prediction+of+Garment+Employees).

### 3.2. Summary of the Dataset
The data is collected from European country at the year of 2015.
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

## 4. IDE and Framework
[Google Colab](https://colab.research.google.com/) is used as the integrated development environment for this project.<br>
The main frameworks utilised in this project were TensorFlow Keras, Pandas, Numpy, Scikit-Learn, and Numpy.<br>

## 5. Methodology
### 5.1. Data Pipeline
Two models, namely the "Sewing Department Productivity Model" (a.k.a. Sewing-Model) and the "Finishing Department Productivity Model" (a.k.a. Finishing-Model), were constructed separately by using the same model pipeline. <br>
The differences between the models all stem from their data pipeline, as summarized as follows:<br>

<table class="center" div align="center">
  <tr>
    <th colspan="3">Data Pipeline</th>
  </tr>                                                          
  <tr>
    <th>Data\Model</th><th>Sewing-Model</th><th>Finishing-Model</th>
  </tr>
  <tr>
    <th>input data</th><td align="center">Sewing department data</td>
    <td align="center">2 $\times$ department training data + sewing department data</td>
  </tr>
  <tr>
    <th>typo error fixed</th><td align="center">Yes</td><td align="center">Same</td>
  </tr>
  <tr>
    <th>wip(work in progress)</th><td align="center">Using original data</td><td align="center">Classified into 3 groups</td>
  </tr>
  <tr>
    <th>date</th><td align="center">Indexed by using chronological order</td><td align="center">Same</td>
  </tr>
  <tr>
    <th>day(weekday)</th><td align="center">Begin with the first consecutive working day in a week(Saturday)</td><td align="center">Same</td>
  </tr>
  <tr>
    <th>quarter(index of week in a month)</th><td align="center">Changed to integers</td><td align="center">Same</td>
  </tr>
  <tr>
    <th>month(/th)<td align="center">Extracted from date</td><td align="center">Same</td>
  </tr>
  <tr>
    <th>day(in months)</th><td align="center">Extracted from date</td><td align="center">Same</td>
  </tr>
  <tr>
    <th>team</th><td align="center">Replaced by the average productivity of the corresponding sewing team</td>
    <td align="center">Replaced by the average productivity of the corresponding departmental team</td>
  </tr>
  <tr>
    <th>train_test_split</th><td align="center">Test_size = 0.2</td>
    <td align="center">Test_size = 0.1 (due to the lack of training data from finishing department)</td>
  </tr>
</table><br>

### 5.2. Models' pipeline 
Both models mentioned above are dense neural network models that are constructed by using the functional API approach.<br>
Below is the summary of the models:<br>
![image](https://user-images.githubusercontent.com/108325848/187822412-7e87e61d-62b6-4aeb-9831-c8d02df22f05.png)

## 6. Result
### 6.1. Sewing Department Productivity Model
Performance of the model:<br>
![image](https://user-images.githubusercontent.com/108325848/187825456-fff0cc2b-6e73-4ab6-825e-a1e5a7c4b00b.png)<br>
![image](https://user-images.githubusercontent.com/108325848/187825574-d4d66107-949c-4db7-9e58-3e7f9f684276.png)<br>

![image](https://user-images.githubusercontent.com/108325848/187819200-d1839d0e-8602-4d72-8530-963648cd29c5.png)<br>

The shaded region is surrounded by a 95% confidence interval, within which we are 95% certain that the **mean value** of prediction lies.<br>
This shall not be confused with the dotted line, which is the 95% prediction interval, within which 95% of our prediction data is contained.

### 6.2. Finishing Department Productivity Model
Performance of the model:<br>
![image](https://user-images.githubusercontent.com/108325848/187825655-aa603b1f-1b59-4981-9cb7-2d397b26d6ab.png)<br>
![image](https://user-images.githubusercontent.com/108325848/187825731-c99476ac-a8c5-48b8-a07a-2d1951081c17.png)<br>

![image](https://user-images.githubusercontent.com/108325848/187712523-078f35c1-7cea-4707-bcdb-a69b0e7a0bd2.png)<br>

## 7. Analysis and Discussion
"All models are wrong, but some are useful."~[George Box](https://en.wikipedia.org/wiki/All_models_are_wrong)<br>

The Sewing-Model is reasonably well fitted, as evidenced by the 95% confidence interval passing through the origin, and the majority of the prediction data converged to a straight line. It also has a validation mean absolute percentage error of 0.029 and a mean absolute percentage error of about 5%. <br> 

The Finishing-Model does not fit well because the 95% confidence interval does not pass through the origin and the majority of the prediction data appears less convergent. Besides, it has a validation mean absolute percentage error of 0.103 and a mean absolute percentage error of about 15%.<br>

Whatsoever, the linear fit in the Finishing-Model indicates that the prediction data increases linearly with the real data. Furthermore, the 0.61 correlation between the prediction data and the real data suggests that the model isn't all that bad. Therefore, it stands to reason that the Finishing-Model can be improved if more data is provided. <br>  

By exploring the data, we know that the main difference between the Finishing-Model and the Sewing-Model is that all "wip" data is missing in the Finishing-Model.
The result of poor fitting in the finising model indicates that "wip" is indeed a crucial variable that is highly correlated to employee productivity in the European garment industry.<br>

We can also check the other attributes/variables in a similar way. By drawing out the targeted variable from the training data, we train the model again, and observe the difference in the accuracy of prediction. <br>

Some may argue that the difference between the results of the Sewing-Model and the Finishing-Model may also be caused by the difference in team attributes(the mean productivity), but I actually did check this argument by using the same mean in both models. The Sewing-Model's prediction accuracy only differs a little, and the model I presented in the data pipeline is in fact marginally more accurate. This forces us to conclude that the missing wip data is the main cause of the inaccuracy of the Finishing-Model, if compared to the Sewing-Model.<br>

Last but not least, if there is an optimal value of wip data, We can check this by generating some lists of sewing data with all attributes fixed but the wip value different. Then we make use of our fairly well fitted sewing data to predict the productivities of each entry in the list. Then we can gather the optimal value from all the lists to find its mean. This mean will be the overall optimal(ideal) value of the wip data.<br>

Why is it advantageous to obtain the optimal value of "wip" (work-in-progress) then? The work-in-progress could be distributed to other teams by the manager, ensuring that most teams have the right amount of work to do and can therefore maximize their production.<br>

The success will roll up and morally encourage the employees to perform better, hence causing the company to win the race against their peers. All of this could stem from one key point: using data mining in their business to determine the optimal value of variables(e.g., work in progress, incentive, overtime) for their employees in order to maximise productivity.<br>

## Conclusion
We have successfully created a good model(the Sewing-Model) with high validation accuracy(validation mean absolute percentage error = 5%, MAE = 0.029) to predict the sewing team productivity in the future. Can this model be used to predict the productivity of all sewing teams in the European garment industry? We can put this question to the test by using data from other sewing teams. One thing is certain: the more sewing team data we collect, the more accurate the model we build.<br>

The Finishing-Model, on the other hand, may be unsatisfactory with a validation mean absolute percentage error of 15%, but as we can see from the graph, it is not completely useless because we are 95% certain that the real data will fall within a certain interval. For example, if the prediction of productivity is 0.2, we are 95% sure that the actual data falls between 0.1 and 0.7, so it can be a good estimate. More data, specifically "wip" data, is required to build a better model.









