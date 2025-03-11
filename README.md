# Model Comparison for Prediction of Heart Failure Death
Evaluates several machine learning algorithms to predict heart failure deaths

Authors: Mason Smith, Arnel Garcesa, Sonia Hernandez, Romney Kellogg, Maddie Molnar


## Introduction 
The project involved using a dataset that tracked death events for patients with heart failure. The dataset has a binary response of DEATH_EVENT due to heart failure. There are ~13 features in the dataset, and the data was collected over 8 months on 299 patients [1]. We are comparing different models to see which one works best on this data set. We are doing this because heart failure affects more 64 million individuals globally, and a predictive model would be helpful to prevent future deaths [3]. As such, we are investigating which model is the best approach to predict if a heart failure patient will die. There are a variety of stakeholders for the models we create, such as insurance companies, hospitals, and researchers. As such, there is clearly a need for these models. 

## Problem Statement 
Doctors need to know if their patients have a higher risk of dying of heart failure. One way to solve this problem is by using machine learning, though what model works best is unclear. As such, five different models were tested to see which one works best: MLP Classifier, Logistic Regression, Decision Tree, K-Nearest Neighbors, and Ordinary Least Squares. From these models, analysis was carried out to see which one had the highest accuracy rate to predict if a patient will die from heart failure or not. Our results can be seen below. 

## Data Description 
The data was collected over the course of 8 months on 299 heart failure patients at the Allied Hospital and Faisalabad Institute of Cardiology in Faisalabad, Pakistan. The patients in this study had medical history with cardiac arrest along with left ventricular systolic dysfunction, and this put them at moderate to severe stages of heart failure according to the New York Heart Association (NYHA) functional classification [1]. Table 1 shows the 13 features in this study along with the target measurements. Anemia, high blood pressure, diabetes diagnosis, smoking and death_event are the boolean values. Creatine phosphokinase, serum creatinine, serum sodium were obtained for this study due to their correlation with heart failure. High creatine phosphokinase levels could be a sign of myocardial injury or stress on the heart that could occur during a heart attack [4]. Serum creatinine levels are usually associated with kidney function, but it has been found that heart failure decreases blood flow to the kidneys and that can reduce kidney function [5]. Serum sodium levels can show the gravity of heart failure as well. Low serum sodium has been seen often in patients that suffer from congestive heart failure. This disease impacts cardiac output through alterations in stroke volume or heart rate [6].


TABLE 1: Meanings, measurement units, and intervals of each feature of the dataset (Chicco & Jurman, 2020). https://doi.org/10.24432/C5Z89R
![image](https://github.com/user-attachments/assets/fe60181b-c651-4265-ba21-d569019fde71)

### Cross Validation
To analyze the proposed predictive models with the limited dataset, a k-folds cross validation was used to validate the models in the face of varying training datasets and the model’s ability to predict on new data. To do this, k=4 folds was selected for the split where each fold had approximately 224 samples where three folds were used as training data and one as a testing set for each cross validation. To ensure that models were sufficiently exposed to both target classes (DEATH_EVENT) sufficiently, a stratified sampling method was used where each fold was composed of 32% of DEATH_EVENT=1 and 68% DEATH_EVENT=0 for both train and testing sets. Implementation of this method used the sklearn.model_selection.StratifiedKFold function. Results are reported as the aggregate performance over all four cross validation sets.

## Models and Data Analysis 
### K-Nearest Neighbors 
The K-Nearest Neighbors model works by calculating the euclidean distance from a test point to the points around it, ranking the relationship between the test point and the other points based on their euclidean distances, and classifying the test point based on the points it is closest to. It is based on the idea that points in the same class will have similar characteristics, and therefore have close euclidean distances.

The decision was made to use K-Nearest Neighbors as a potential solution to this problem because the problem works well as a classification problem, since the target, the death rate, is binary. On top of this, patients that will face death from heart failure likely have similar characteristics, so any test datapoint being classified using K-NN will be close to the training points that indicate heart failure. 

The model was implemented using the Scikit-Learn neighbors library to get the associated K-NN model. The code then iterated through 4 different K-Values 3, 5, 10, and 15. The first three came from assignment #2, while the last one was chosen due to the previous value I was using, 20, giving nearly the same result as 10. To show the accuracy, a confusion matrix and accuracy score was calculated. From these, it could be seen that the K value of 10 had the best accuracy value at 66%. Interestingly, though, the classification accuracy dropped when k=15. This was likely due to the distance calculation between the 15 neighbors, and having multiple close points of different classes.  

![image](https://github.com/user-attachments/assets/dbf2cee9-37ee-46f8-82ec-b8d74e9fe01c)

Figure 1. Confusion Matrix for K-NN Classification when k=10

### Multilayer Perceptron
Multilayer perceptrons (MLP) is an extension of the standard perceptron where additional hidden layers are added to capture non-linear discriminants. The network is fully connected where x and bias is fed to the input layer and propagates throughout each hidden layer eventually arriving at the output layer. Each hidden layer additionally has a bias node. Since this problem has k=2 classes the output layer has a single node determining y. To optimize the model, the module sklearn.model_selection.GridSearchCV was used to search over possible model parameters. The space of parameters that was searched can be found in Table 2. All input data for train and test sets were scaled using sklearn.preprocessing.StandardScalar.



TABLE 2: MLP Grid Search
![image](https://github.com/user-attachments/assets/7099326b-2257-479f-938f-5a6af532a139)

![image](https://github.com/user-attachments/assets/60f32111-8172-4ef9-b203-5934d8ed6d99)

Figure 2.  Confusion matrix for Multilayer Perceptron

MLP had an prediction accuracy of 83% ± 3% across all four cross validation sets. The confusion matrix for MLP can be seen in Fig. 1 where it was more accurate in classifying Lived (DEATH_EVENT=0) classes with 90% accuracy of Lived predictions whereas it was only 69% accurate in classifying Dead predictions.  

![image](https://github.com/user-attachments/assets/9580d068-1631-4144-ba59-5d0e248640e2)

Figure 3.  Receiver Operating Characteristic for Multilayer Perceptron 
model with an Area Under the Curve of 0.88.

The receiver operating characteristic (ROC) curve for MLP can be found in Fig. 2. Here, we can see that MLP does a fairly good job at classifying at different decision thresholds as indicated by the area under the curve (AUC=0.88). 

### Logistic Regression
Logistic Regression is a classifier in ML that uses the sigmoid function to convert the output into a probability score of 1 or 0. This model is typically used for classification problems. This algorithm calculates the weights for each independent variable and aggregates them to predict the probability of the outcome. The binary target in this dataset was “death rate”, and the logistic regression model was tested to observe its prediction accuracy on this outcome. Jupyter Notebook was used to write the python code and the sklearn.linear_model.LogisticRegression library was used to develop the Logistic Regression model. The accuracy of the model earned a score of 81%. Fig. 3 demonstrates the confusion matrix for the logistic regression, the results show 183 true positive values for class 0 which indicates the patient lived, and it predicted 60 true positive values for class 1 which is when the patient died during the follow-up period.

![image](https://github.com/user-attachments/assets/86f2dd63-7eb8-488f-ba42-dcda9acfe56a)

Figure 4. Confusion matrix for Logistic Regression

The Receiver Operating Characteristic (ROC) curve seen in Fig. 4 was used to demonstrate the performance of the logistic regression model. The true positive rate is plotted against the false positive rate to create the ROC curve. The area under the curve is 0.86. This can tell us the model is fairly good at classification, but it may not be as precise on its own to predict whether a patient can survive from heart failure with the given dataset.

![image](https://github.com/user-attachments/assets/fd7e1683-a768-4f62-a36b-b454d58d156b)

Figure 5. Receiver Operating Characteristic for Logistic Regression 
model with an Area Under the Curve of 0.86.


### Decision Tree
A Decision Tree is a flow-chart structure that uses a non-parametric supervised learning method mainly used for classification and regression use cases. Decision trees create predictions of the target variable by learning decision rules from the data features. Decision trees are composed of the following components: Root Nodes, Decision/Internal Nodes, Branches, Leaves, Splitting, and Decision Criteria. The Root Node is the first or top node in the tree, this node contains the input or the initial decision of the tree. The Decision/Internal Nodes are the decisions that are created from specific features and attributes from the data features. Branches are the pathways that connect the Decisions Nodes and often denote the outcome of the Decision Node (e.g. yes/no). Leaves are the final nodes in the Decision Tree and represent the final predictions, they are typically found at the bottom of the Decision Tree. Splitting occurs when a node needs to be further divided into two or more sub-nodes. The Decision Criteria is the rules that accompany each Decision Node. Because Decision Trees can be visualized, it makes them easy to understand and easy to interpret. They are also easy to validate with statistical tests, therefore making them a fairly reliable method. 

A Decision Tree was used for this study, it was trained on a dataset split across four k folds with the intention to accurately predict a DEATH_EVENT with the test data. The Decision Tree as seen in Fig. 5 was created in a Python Jupyter Notebook and utilized the Scikit-Learn DecisionTreeClassifier toolbox. A confusion matrix was created as seen in Fig. 6 to measure the accuracy of the model’s ability to predict the target DEATH_EVENT on the test data and had an overall accuracy of approximately 79%. The Decision Tree also uses impurity reduction to calculate the feature importance. As seen in Table 3, the feature importance was ranked from most to least. The top three most important features were Time (55%), Serum Creatinine (12%), and Ejection Fraction (10%). This indicates that the time between follow up appointments had the most significance on the target outcome. The features with the least amount of importance were Smoking, Sex, Diabetes, and Anaemia (0%). This means that regardless if the patient smoked, the sex of the patient (woman/man), if the patient had diabetes, if the patient's red blood cells/hemoglobin decreased were irrelevant to affecting the target outcome.  

![image](https://github.com/user-attachments/assets/c9047946-0ae3-4f8d-a3fe-c93d563cbc01)

Figure 6. Decision Tree to determine DEATH_EVENT in patients suffering from heart failure.

![image](https://github.com/user-attachments/assets/dc7816ff-a5e1-4292-8760-50f7b499bdf8)

Figure 7. Confusion matrix for Decision Tree.

### Linear Discriminant Analysis
Linear Discriminant Analysis (LDA) is a technique which projects feature values of a dataset onto a new axis or plane in order to achieve the best separation between two or more classes. This separation is achieved by attempting to maximize the distance between the centers of different classes while minimizing the variation within each individual class. This assumes instances of a class are linearly separable from instances of other classes. This is a discriminant-based approach where knowledge about the densities is not required. In LDA, it is assumed the discriminant function is linear in the features. 
In this study, LDA was conducted across the four folds to fit the model then make a prediction of DEATH_EVENT upon the test dataset. The predictions are then compared with the ground truth label. In each of the four folds, the accuracy rate ranged from 78.7% to 89.3%. The aggregate confusion matrix is presented below in Fig. 8.

![image](https://github.com/user-attachments/assets/4d2ec80c-16dd-4dab-b890-fd0d11478f6e)

Figure 8. Confusion matrix for Linear Discriminant Analysis

When the predictions are aggregated among the four folds, the overall accuracy rate is 83%. The second fold was the one with the highest accuracy rate among the four folds. Information from that case is shared below. 

![image](https://github.com/user-attachments/assets/1f8a3690-b481-407a-a88a-e95d82f40830)

Figure 9. Separation of Classes for Fold 2

The training of the LDA can be visualized in Fig. 8 where dots indicate the linear classifier (boundary) was successful in separating the two classes. The X’s indicate cases which are not successfully classified.
The coefficients from Fold 2 are also output below in Fig. 9 .



## Performance Analysis 
The K-Nearest Neighbors model likely did poorly due to the nature of the data. While the data is well suited for classification, the issue seems to be that the K-NN model is incorrectly classifying death events as non-death events. This may be due to the nature of the data, and those incorrectly classified points being closer in distance to the non-death events due to their features. This issue can be seen by how the number of false positives increases as the k-value increases, as the test points are now being compared against more points. As such, this model is not the best for this problem. 

While Logistic Regression is a great model to be used for binary classification, the dataset obtained for this study is fairly small and this could have caused some underfitting in this model. Having a larger dataset could help identify intricacies in the relationship between predictors and outcomes. As seen in Fig.10, logistic regression obtained an accuracy rate of 81%. The heart failure dataset contains 203 instances labeled as class 0 while 96 were labeled as class 1, and the logistic regression model was able to correctly classify 183 as class 0 and 60 as class 1.

Multilayer perceptron provides an opportunity to capture nonlinear relationships between variables and has the advantage of providing a probabilistic prediction similar to logistic regression with similar prediction accuracy. However, MLP in particular is susceptible to overfitting when overtrained and has no interpretable information to glean from the trained model. 

Decision Tree allows easy interpretation and therefore a closer look into what attributes could be having the greatest impact on patient health and survival rate. By reviewing the Decision Tree, the top three most used features were identified. These features were Time (55%), Serum Creatinine (12%), and Ejection Fraction (10%). This allows medical professionals to be able to gain further insight to what features could be playing a significant role in their patients health and likelihood to survive. Therefore allowing potentially more informed and curated care to be given to their patients. The Decision Tree had an accuracy rate of approximately 79% percent. This accuracy rating could be due to the Decision Tree creating an over complex tree (overfitting) due to the relatively high number (12) of features used in this dataset. Additionally, the training dataset could have been too small, which would result in the model to not be able to accurately capture the underlying patterns within the data. However, despite the level of accuracy, because the Decision Tree provides a visual representation, and measures the feature importance, medical professionals can easily interpret the model and can potentially have an easier time identifying inaccuracies and how those inaccuracies came to be. 

LDA was among the best performing models in predicting DEATH_EVENT from the methods considered. This is despite a minority of cases which are incorrectly classified based on their position compared to the hyperplane. This inaccuracy may be due to the close nature of the points which are already separated as best as possible and thus leads to overlap between each class. One potential reason for the strong performance is the fact that all features under consideration are numeric. This may enable the projection from the features under consideration onto the revised axis. Furthermore, when the coefficients of the hyperplane are fitted from the training data, these can be applied to the continuous and binary features as necessary. Ultimately, a health researcher can utilize such coefficients to glean an initial understanding of the magnitude and direction of certain features upon the outcome of DEATH_EVENT. Finally, a fitted model generated by LDA offers a visualization which can provide a quick glance at what class would be expected to be classified based on input data. 

![image](https://github.com/user-attachments/assets/ed57f516-98f0-49be-8e2e-d54af0337594)

Figure 10. Confusion Matrix and Accuracy Results


## Conclusion
Based on the presented results, we have two recommendations for which models would likely provide the best implementation in a clinical setting. The first recommendation is conditional on if a clinician were to desire an interpretable model. In this case, we would recommend decision tree or linear discriminant analysis due to the feature weights hinting at which risk factors most determine the death event. This may provide insight on preventive measures in the future. These two models provide similar prediction accuracy so further validation would be needed to determine which is best in the interpretable models. The second recommendation is conditional on if a clinician would like a probabilistic prediction of the death event. In this case, we would suggest the multilayer perceptron or logistic regression which both had similar prediction accuracies. In general, we would not suggest k-nearest neighbors due to the low prediction accuracy. 

In general, these models suggest that machine learning can provide value in a clinical setting like this to predict and possibly prevent heart failure. Consequently, a provider may be enabled to provide better patient care with an accurate model. However, we believe that approximately 80% accuracy across all of these models is still fairly low to base life and death decisions on. Therefore, we caution implementation of these models as direct prediction mechanisms in such a critical task.

## References:

[1] Chicco D, Jurman G. Heart failure clinical records. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5Z89R.

[2] K-Nearest Neighbor. (2021). Medium. https://medium.com/swlh/k-nearest-neighbor-ca2593d7a3c4 

[3] Savarese G, Becher PM, Lund LH, Seferovic P, Rosano GMC, Coats AJS. Global burden of heart failure: a comprehensive and updated review of epidemiology. Cardiovasc Res. 2023 Jan 18;118(17):3272-3287. doi: 10.1093/cvr/cvac013. Erratum in: Cardiovasc Res. 2023 Jun 13;119(6):1453. PMID: 35150240.

[4] Zahler D, Rozenfeld KL, Merdler I, Itach T, Morgan S, Levit D, Banai S, Shacham Y. Relation between Serum Creatine Phosphokinase Levels and Acute Kidney Injury among ST-Segment Elevation Myocardial Infarction Patients. J Clin Med. 2022 Feb 21;11(4):1137. doi: 10.3390/jcm11041137. PMID: 35207410; PMCID: PMC8877638.

[5] Maulion C, Chen S, Rao VS, Ivey-Miranda JB, Cox ZL, Mahoney D, Coca SG, Negoianu D, Asher JL, Turner JM, Inker LA, Wilson FP, Testani JM. Hemoconcentration of Creatinine Minimally Contributes to Changes in Creatinine during the Treatment of Decompensated Heart Failure. Kidney360. 2022 Apr 18;3(6):1003-1010. doi: 10.34067/KID.0007582021. PMID: 35845336; PMCID: PMC9255871.

[6] Abebe TB, Gebreyohannes EA, Tefera YG, Bhagavathula AS, Erku DA, Belachew SA, Gebresillassie BM, Abegaz TM. The prognosis of heart failure patients: Does sodium level play a significant role? PLoS One. 2018 Nov 8;13(11):e0207242. doi: 10.1371/journal.pone.0207242. Erratum in: PLoS One. 2019 Sep 19;14(9):e0223007. PMID: 30408132; PMCID: PMC6224129.

[7] American Kidney Fund. (2023, November 10). Serum creatinine test. https://www.kidneyfund.org/all-about-kidneys/tests/serum-creatinine-test#:~:text=Your%20serum%20creatinine%20level%20is,creatinine%20with%20a%20urine%20test





