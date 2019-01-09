# Machine Learning Engineer Nanodegree
## Capstone Project
Jinchuan Wei 

Janunary 1st, 2019

## I. Review

### Project Overview
In this project, we are going to predict logerror of zillow price which is defined as the following:
![logerror](logerror.png)

We have data set which has >= 50 columns of features and >= 90000 data samples, so we need to divide this data set into training data set and test data set so that we can train and test out model 

### Problem Statement
In order to solve the problem above, we need to try several models, and choose the best model to set it as our final solution. Before working on testing our model, clearing data is also important in this case, since we have much noise in our data set including missing data and some miss-calculated data. Then, firstly, we can try some simple regression models such as linear regression and polynomial regression, and use mean square error to see whether it has reasonable performance. After that, we can improve our model by using some more complex models such as decision/random regression tree algorithms. After that, adding normalization to avoid over-fitting is also important in this case. In order to avoid over-fiting, in addition to use some normal methods such as L1/L2 norms, we can try ensemble methods such as bagging, cross-validation method, adaboost and gradient-boosting algorithms to try! Finally, we find both of adabooting decision regression and gradient boosting tree regression algorithms can have good perforamance compared to pure regression algorithms such as decision regression algorithm. 

### Metrics
In this project, we need to use mean square error to our metrics. If mean square error is smaller than or equal to 0.02, it should be a good model, otherwise we need to improve it! Also, we need to draw figures to see whether our models' predictions are reasonable! 


## II. Analysis

### Data Exploration
In this project, the data set is huge, and we need to handle it very carefully. Firstly, since the data sample has over 50 features, some of them are lacking huge amount of data. In this case we have two options, either we ignore those features who have huge amount of missing data, or we can just fill those missing data with mean value of the other unmissed data. In this project, we tried the second option, which means that we fill missing data with mean values of those unmissed data. Then, what we need to is that we explore the distribution of logerror we need to predict, and it turned out that it looks like a guassian function. After that, we want to study the correlations between every feature and logerror. That's because higher correlations indicate that feature is more related to or more influential to logerror predictions. If the correlation of some feature and logerror is near zero, it means that that feature has nealy no effects on out logerror predictions. Using this way, we can just igore those feactures having lower correlations, and use features having higher correlations to train out model, which can save a huge amount of time and increase model's performance in some cases  

### Exploratory Visualization
Here is the logerror distribution figure:

![logerror_figure](logerror_figure.png)

Here is correlations between features and logerror:

![correlations](correlations.png)

### Algorithms and Techniques
#### method 1: decision tree regression
In this method, we first try a simple algorithm for predicting our logerror values. decision tree algorithm basically to choose most optimized feature to decide which category we need to choose so that we can get the most desired predition. Finally. we set the max-depth as 20 since we have over 50 features, but we cut off those unrelated features based on correlations and leave only 18 features in our columns features. Finally, we got a mean square error which is 0.021273486707590866, which is not very satisficatory. But, it is a good start! Here is what looks like when I sample some test data and compare it with my model's predictions

![decison_tree](figure1.png)

#### method 2: decsion tree plus adaboost algorithms:
In this case, we need to add ensemble method so that we can improve those models with bad performance and combine with models with good performace. For one thing, it can improve our final model's performance. For another thing, it can avoid overfitiitng. The final mean-saure error is 0.016951678638327763, which is good! 
![decison_tree_adaboost](figure2.png)

#### method 3:
In this case, we can use gradient boosting tree regression algrithm to improve our pure tree regression algorithm. The basical idea behind this algorithm is that we need to improve current model based on current difference bwteen current model's predictions and true logerror values. The issue for this algorithms, it can be over-fitting in some sense, but if we can carefully control the number of steps, it can significally improve the pure tree regression algorithms, the best model of this algorithm we tested has mean square error 0.01599242456077152.

![gradient_boosting](figure3.png)


## III. Methodology

### Data Preprocessing
As for data processing, as we talked above, firstly, we need to find all of features who have missing data since we couldn't train our model if we have missing data in our data set. There are two options, the first option will be that we can just ignore those features having missing data, but it can remove some valuable information of our data set. The second option is that we can fill those missing data based on mean values of the other unmissing data, which is more reasonale in this case. Thus, we choose the second option to handle data processing in out project. After that, we need to investigate how those features are related to logerrors we need to predict. In this case, we normally use correlations to determine whether that feature is correlated to our logerror. If the correlation between logerror and feature is near zero, it means that that feature is not correlated to out logerror we need to predict. If that correlation of some feature is close to 1, it means that feature has high correlations with logerror! Thus, based on the correlations we compute, we can get the desired features we need to select to train our model, which can save our training time! 

### Refinement
In this project, we firstly tried several simple regression algorithms such as linear regression, polynomial regression, support vector regression and decision tree regression, and we find that most of them don't satisfy our requirements. For one thing, they have very high test mean square error, and the predicted results don't match with logerror we need to predict! There are several reasons why they don't work very well. As for linear regression, the algorithm model complexity is very small, and we won't be able to use it to resolve some complex data set in this case. As for polynomial regression, in some sense, it can solve some disadvantages of linear regression, but it has huge disadvantage which is that it can be over-fitting. Thus, even we can have low training mean square error, but it is hard to control the model's degree to avoid overfitting. Thus, the flexibility of polynomial is that great! As for support vector regression, it has some limitaions. In this project, it doesn't have very obvious hyper-plane to divide data set, so it doesn't have good result in this case. Probably, we can try kernerl SVR, but we didn't try it. As for decision tree regression, since it can be used to work out in this project, and it does have good performance, but we couldn'ty make it satisfy our requirement no matter how to set parameters such as max-depth, learning rate. Then, we consider about add adaboost and gradient boosting to improve pure regression algorithm. Since tree regression algorithm has best performance among all of regression algorithms we tried, so we decide to add adaboost and gradient boosting algorithm on tree regression algorithm. It turned out that we improved tree regression algorithm's performance after adding ensemble method. As for adaboost algorithm, it balance between different models so that we can get the best model!  


## IV. Results

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

