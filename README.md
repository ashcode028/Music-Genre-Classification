# Music-Genre-Classification

This was made as a part of Machine Learning course at IIIT,Delhi. [Link to blog](https://medium.com/@ashitaboyina/music-genre-classification-70ae70469403)

## Dataset Description
We are using the __GTZAN__ dataset which contains a total of 1000 audio files in .wav format divided into 10 genres. Each genre has 100 songs of 30-sec duration. Along with the audio files, 2 CSV files containing features of the audio files. The files contain mean and variance calculated over multiple features that can be obtained from an audio file. The other file has the same composition, but the songs were divided into 3-second audio files. 

![Genres](https://github.com/ashcode028/Music-Genre-Classification/blob/7aff7f3c06156814dca7298baec67d6cc87df4cc/Features_extracted_plots/Genre.png)
### Audio signal feature extraction:
We convert every audio file to signals with a sampling rate to analyze its characteristics. 
Every waveform has its features in ttwo forms:
- Time domain- nothing much information about music quality can be extracted and explored apart from visual distinction in the waveforms
![](https://github.com/ashcode028/Music-Genre-Classification/blob/d9012c9e43e4cd18639841b931624ee19292352d/Features_extracted_plots/z1.png)
![](https://github.com/ashcode028/Music-Genre-Classification/blob/d9012c9e43e4cd18639841b931624ee19292352d/Features_extracted_plots/z2.png)
- Frequency domain which we get after fourier transform of two types: Spectral features and Rhythm features
#### Spectral
![MFCC](https://github.com/ashcode028/Music-Genre-Classification/blob/ef966977e66f7f62a116ffb6c3b2a2a4f3f3f8cc/Features_extracted_plots/mfcc1.png)
![Spectogram](https://github.com/ashcode028/Music-Genre-Classification/blob/ef966977e66f7f62a116ffb6c3b2a2a4f3f3f8cc/Features_extracted_plots/stft_spectrogram2.png)
#### Rhythm features
![](https://github.com/ashcode028/Music-Genre-Classification/blob/ef966977e66f7f62a116ffb6c3b2a2a4f3f3f8cc/Features_extracted_plots/rhythm1.png)


MFCC and Rhythm feature plots provide a matrix based information for the unique features. Both the features have been mapped with the duration of the music file.

### Preprocessing
After extraction of features, all columns were not null. So extra values were not added.
Why is it important to preprocess the data?
- The variables will be transformed to the same scale.
- So that all continuous variables contribute equally 
- Such that no biased results 
- PCA very sensitive to variances of initial variables
 If large variance range difference between features , the one with larger range will dominate
- The boxplots of each feature shows some features have very large differences in their variances.
- PCA with both normalisation(minMaxScaler) and standardisation(StandardScaler) is done and difference noted.
 
 ## Methodology
 Feature extraction -> correlation matrix -> PCA
 - With 30 secs sample
 ![](Features_extracted_plots/Corr_Heatmap30_better.png)
 ![](https://github.com/ashcode028/Music-Genre-Classification/blob/ef966977e66f7f62a116ffb6c3b2a2a4f3f3f8cc/Features_extracted_plots/PCA30_std.png)
 - With 3 secs sample
![](Features_extracted_plots/Corr_Heatmap3.png)
![](https://github.com/ashcode028/Music-Genre-Classification/blob/ef966977e66f7f62a116ffb6c3b2a2a4f3f3f8cc/Features_extracted_plots/PCA_std.png)
- Less outliers/ variance for some classes found in principal components:


![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/Features_extracted_plots/pc1.png)
![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/Features_extracted_plots/pc2.png)


#### Inferences till this step:
- pca.explained_variance_ratio_=[0.20054986 0.13542712]
Shows pc1 holds 20% percent of the data, pc2 holds 13% of the data
- Big clusters of metal , rock, pop ,reggae, classical can be seen.
- Jazz ,country, are separable to one extent.
- Hip-hop,disco,blues are very dispersed and can’t be seen 
- Majority are easily separable classes
- Decided to proceed to modelling phase by using 3 sec sampled feature set with standardization as it aggregated the genres into more linearly separable clusters than normalisation

## Classification:
### Logistic
This model is a predictive analysis algorithm based on the concept of probability. GridSearchCV was used to pass all combinations of hyperparameters one by one into the model and the best parameters were selected. 

Without Hyperparameter tuning:

![Confusion Matrix - Logistic Regression Base Model](https://user-images.githubusercontent.com/61936574/143768349-137e14a6-cfa1-4f38-8a31-a7806dcde571.png)
![ROC Curve Base Model](https://user-images.githubusercontent.com/61936574/143768389-584ab388-b0ed-4a60-bbb9-205e3179ad79.png)

|Metric          |  Value                        |
|:---            |                           ---:|  
|Accuracy score|0.67267|
|Precision|0.74126|
|Recall|0.74098|

Using Hyperparameter tuning:

![Confusion Matrix - Logistic Regression After HT](https://user-images.githubusercontent.com/61936574/143768481-3a4fd0be-2161-4ee0-b28f-13486507d51a.png)
![ROC Curve after HT](https://user-images.githubusercontent.com/61936574/143768497-3cb355c0-3516-4774-9c9f-ce9df47991b8.png)

|Metric          |  Value                        |
|:---            |                           ---:|  
|Accuracy score|0.70504|
|Precision|0.70324|
|Recall|0.71873|

### SGD Classifier
Took SGD as baseline model and performed hyperparameter tuning for a better performance.Though difference werent that great even after HP tuning.

Without Hyperparameter tuning:

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_NB_KNN_SGD/SGD.png)

|Metric          |  Value                        |
|:---            |                           ---:|  
|Accuracy score| 0.6126126126126126|
|Precision|0.6142479131341332|
|Recall|0.6172558275062101|


With Hyperparameter tuning:

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_NB_KNN_SGD/GS_SGD.png)
![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_NB_KNN_SGD/ROC_SGD.png)

|Metric          |  Value                        |
|:---            |                           ---:|  
|Accuracy score| 0.6441441441441441|
|Precision|0.6386137102787109|
|Recall|0.6421140902032518|


### Gaussian NB

We used a Simple Naive Bayes classifier, one vs Rest Naive Bayes as baseline models.
Then used Hyperparameter testing to get better performance.

Without Hyperparameter tuning:

![ ](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_NB_KNN_SGD/simpleNB.png)

![ ](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_NB_KNN_SGD/ROC-simpleNB.png)


|Metric          |  Value                        |
|:---            |                           ---:|  
|Accuracy score|0.48598598598598597|
|Precision|0.4761542269197442|
|Recall|0.4902979078811803|


With Hyperparameter tuning:


![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_NB_KNN_SGD/GS_NB.png)

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_NB_KNN_SGD/ROC_NB.png)
 
|Metric          |  Value                        |
|:---            |                           ---:|  
|Accuracy score|0.5155155155155156|
|Precision|0.49864157768533374|
|Recall|0.5050696700999591|

### KNN
This model almost outperformed compared to Gaussian NB models. 
As we can see , after HP tuning , correlation between the features has decreased, some had even 0 correlation.

Without Hyperparameter tuning:

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_NB_KNN_SGD/KNN.png)
![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_NB_KNN_SGD/ROC_simpleKNN.png)

|Metric          |  Value                        |
|:---            |                           ---:|  
|Accuracy score|0.8603603603603603|
|Precision|0.8594536380364758|
|Recall|0.8583135066852872|

Using hyperparameter tuning :

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_NB_KNN_SGD/GS_KNN.png)
![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_NB_KNN_SGD/ROC_KNN.png)

|Metric          |  Value                        |
|:---            |                           ---:|  
|Accuracy score|0.9059059059059059|
|Precision|0.9073617032054686|
|Recall|0.905944266718195|

Best params:
{'metric': 'manhattan', 'n_neighbors': 1, 'weights': 'uniform'}


### Decision Trees
- Took DT as baseline model which didnt give great results, with accuracy around 65%.

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/Plots_DT_ENSEMBLE/DT-ROC.png)

|Metric          |  Value                        |
|:---            |                           ---:|  
|Accuracy score| 0.637758505670447|
|Precision|0.6396387192624916|
|Recall|0.6376582879474517|

- Used ADA boosting which reduced the performance(rock,pop,disco)

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/Plots_DT_ENSEMBLE/ADA-ROC.png)

|Metric          |  Value                        |
|:---            |                           ---:|  
|Best parameters| n_estimators=100|
|Accuracy score| 0.5010006671114076|
|Precision| 0.48730102839842837|
|Recall|0.4992406459587978|

- Then gradient boosting which increased the accuracy exponentially.

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/Plots_DT_ENSEMBLE/GRADIENT-ROC.png)

|Metric          |  Value                        |
|:---            |                           ---:|  
|Best parameters| n_estimators=100|
|Accuracy score| 0.8238825883922615|
|Precision| 0.8266806080093154|
|Recall|0.8232200760446549|

- CatBoost was having high AUC for all genres unlike gradient which had low accuracy for some genres

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/Plots_DT_ENSEMBLE/CATBOOST-ROC.png)

- Cat boost outperformed ensemble methods. Gradient boost was close enough with 82% accuracy, rest all were in between 50-60%

|Metric          |  Value                        |
|:---            |                           ---:|  
|Best parameters| loss function:”Multiclass”|
|Accuracy score| 0.8972648432288192|
|Precision| 0.8979267969111706|
|Recall|0.8972734276109252|

### Random Forest
- As shown here RF was having around 80% accuracy but XGB boosting reduced the accuracy to 75%

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/Plots_DT_ENSEMBLE/RF-ROC.png)

|Metric          |  Value                        |
|:---            |                           ---:|  
|Best parameters |n_estimators=1000 max_depth=10|
|Accuracy score  |0.8038692461641094             |
|Precision       |0.805947955999254              |
|Recall          |0.8026467091527609             |

- Cross Gradient Boosting on Random Forest reduced the accuracy , it even reduced precision ,recall to large extent.

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/Plots_DT_ENSEMBLE/XGB-RF-ROC.png)


|Metric          |  Value                        |
|:---            |                           ---:|  
|Best parameters |objective= 'multi:softmax'|
|Accuracy score  | 0.7505003335557038            |
|Precision       |     0.7593347049139745     |
|Recall          |0.7494976488750396           |

### XGB Classifier
- Correlation matrix shows there is very less correlation among variables

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/Plots_DT_ENSEMBLE/XGB.png)
![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/Plots_DT_ENSEMBLE/XGB-ROC.png)

- Best performed model among all DT and RF models with every genre was classified with atleast 85+% accuracy 
- Genres like classical,hiphop had even 100% accuracy
- XGBoost improves upon the basic Gradient Boosting Method framework through systems optimization and algorithmic enhancements.
- Evaluations


|Metric          |  Value                        |
|:---            |                           ---:|  
|Best parameters |learning rate:0.05, n_est =1000|
|Accuracy score  |0.9072715143428952            |
|Precision       |0.9080431364823143             |
|Recall          |0.9072401472896423            |

### MLP
This model is an Artificial Neural Network involving multiple layers and each layer has a considerable number of activation neurons. The primary training included random values of hyperparameters except the activation function . This initiation reflected overfitting in the data for different activation functions : <br>

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_MLP/activation.jpg)

|Activation |  Training Accuracy  |  Testing Accuracy  | 
|:---       |                 ---:|                ---:|
|relu       |0.9887142777442932   |0.5206666588783264  | 
|sigmoid    |0.941428542137146    |0.4970000088214874  |
|tanh       |0.9997143149375916   |0.49266666173934937 |
|softplus   |0.9991428852081299   | 0.5583333373069763 |

From the following graph, we choose softplus to be the best activation function, considering softmax to be fixed for output <br>
Upon looking the graph, we can conclude a very high variance in testing and training accuracy and so we know that our model is overfitting. In fact the testing loss starts to increase which indicates a high cross entropy loss. This will be dealt later. For now we see that softplus, relu and sigmoid, all 3 have performed similar on training and testing set thus we will go with softplus since it provides a little less variance than others.

#### Hyperparameter tuning has been done manually by manipulating the following metrics: 
- Learning rate <br>
activation = softmax <br>
no. of hidden layers = 3; neurons in each = [512,256,64] <br>
activation of output layer is fixed to be softmax epochs = 100 <br>

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_MLP/learningRate.jpg)

|Learning Rate |  Training Accuracy  |  Testing Accuracy  | 
|:---          |                 ---:|                ---:|
|0.01          |0.4044285714626312   |0.335999995470047   | 
|0.001         |0.9888571500778198   |0.5666666626930237  |
|0.0001        |0.9684285521507263   |0.5513333082199097  |
|0.00001       |0.7134285569190979   |0.4996666610240936  |

From the above graphs we see that 0.01 definitely results in over convergence and bounces as reflective from the accuracy graph. 0.001 has a very high variance and loss increases margianally with low acuracy so it isn't appropriate as well. <br>

The best choice for alpha is either 0.0001 or 0.00001. <br>
0.00001 has a relatively low variance and loss converges quickly with epochs but accuracy on training and testing set is pretty low. <br>
0.0001 has a better performance but variance is very high

- no.of hidden layers <br>
activation = softmax <br>
learning rate = 0.0001 <br>
activation of output layer is fixed to be softmax epochs = 100 <br>

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_MLP/layers.jpg)

|Number of layers|  Training Accuracy  |  Testing Accuracy  | 
|:---            |                 ---:|                ---:|
|2               |0.9782857298851013   |0.5383333563804626   | 
|3               |0.9869999885559082   |0.5443333387374878  |
|4               |0.9921428561210632   |0.5506666898727417  |

In conclusion, increasing or decreasing the number of layers have no effect on variance. This is because we have too many neurons per layer. So we take 3 layers and reduce the number of neurons. 

- Number of neurons <br>
activation = softmax <br>
learning rate = 0.0001 <br>
number of layers = 3 <br>
activation of output layer is fixed to be softmax epochs = 100 <br>
drop out probability = 0.3 <br>
alpha = 0.001<br>

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_MLP/neurons.jpg)

|Number of neurons|  Training Accuracy  |  Testing Accuracy  | 
|:---             |                 ---:|                ---:|
|[512, 256, 128]  |0.9984285831451416   |0.563666641712188   | 
|[256, 128, 64]   |0.915142834186554    |0.5149999856948853  |
|[180, 90, 30]    |0.7991428375244141   |0.503000020980835   |
|[128, 64, 32]    |0.6991428732872009   |0.4900000095367431  |

Now for the same neuron set, we apply regularization and neuron dropout to find any change in the variance for high number of neurons with reducing the number of neurons <br>

- regularization and decomposition <br>

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_MLP/regularization.jpg)

|Number of neurons|  Training Accuracy  |  Testing Accuracy  | 
|:---             |                 ---:|                ---:|
|[512, 256, 128]  |0.6759999990463257   |0.5830000042915344  | 
|[256, 128, 64]   |0.5278571248054504   |0.5189999938011169  |
|[180, 90, 30]    |0.43642857670783997  |0.4629999995231628  |
|[128, 64, 32]    |0.386428564786911    |0.4203333258628845  |

So in conclusion, if we have high number of neurons per layer, then applying regularization techniques will increase the accuracy and decrease the variance overall. If we do not apply any regularization techniques then we can have moderate number of neurons to have a decent accuracy on training and testing set with low accuracy. <br>

##### For our purposes, we select high number of neurons per layer with regularization
#### Final MLP model
From all our analysis and extra experimentation we conclude our model with following metrics: <br>
- activation : softmax 
- learning rate : 0.0001
- number of hidden layers = 3
- number of neurons in each layer = [512,256,128]
- epochs = 100
- regularization and dropout true

![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_MLP/heatmap.png)

Precision on the model :	 0.5774000692307671 <br>
Recall on the model : 		 0.583 <br>
F1score on the model : 		 0.5801865223684216 <br>
Accuracy on the model : 	 0.6130000042915345 <br>


![](https://github.com/ashcode028/Music-Genre-Classification/blob/main/PLOTS_MLP/roc.png)

Even after hyperparameter tuning, the best accuracy is just above 60%. The reason is simply because of overfitting and underperformance due to inability to pick up each feature. This creates amazing accuracy on the training set but always misses out on the testing set.


### SVM
This model outperformed every other model and gave the best accuracy. Manual hyperparameter tuning was done. Linear, polynomial and RBF  kernel were compared using confusion matrix.

#### Best Linear Kernel Model: 
![Best SVM linear kernel](https://user-images.githubusercontent.com/61936574/143769002-09ed2434-fa45-4b84-860e-0648523a3e59.png)
![Plot of Classification Report - SVM linear Best](https://user-images.githubusercontent.com/61936574/143769447-9549a82c-455e-42d3-bb12-a4566aaa5854.png)

|Metric          |  Value                        |
|:---            |                           ---:|  
|Best parameters |C=1.0,kernel='linear',random_state=0|
|Accuracy score  |0.70672342343265456            |
|Precision       |0.7180431364823143             |
|Recall          |0.71234655872896242            |

#### Best Polynomial Kernel Model: 
![Best SVM poly kernel of degree 7](https://user-images.githubusercontent.com/61936574/143769340-5b7fb30a-9f11-4f8a-9dbd-67f93442a1b7.png)
![Plot of Classification Report - SVM Poly Best](https://user-images.githubusercontent.com/61936574/143769525-c0ea5e9e-4041-4594-ac49-254a59f075c4.png)

|Metric          |  Value                        |
|:---            |                           ---:|  
|Best parameters |C=1.0,kernel='poly',degree=7|
|Accuracy score  |0.88242715143428952            |
|Precision       |0.8780431364823143             |
|Recall          |0.87035601472896557            |

#### Best RBF Kernel Model: 
![Best SVM rbf kernel c=200 gamma=4](https://user-images.githubusercontent.com/61936574/143769535-e595601a-8465-4913-92cb-883c62172829.png)
![Plot of Best Classification Report - SVM Best RBF](https://user-images.githubusercontent.com/61936574/143769548-aea05e19-5ac8-4d4e-bf4e-2574fe99d572.png)

|Metric          |  Value                        |
|:---            |                           ---:|  
|Best parameters |C=200,kernel='rbf',gamma=4|
|Accuracy score  |0.9424715143428952            |
|Precision       |0.939297323879391             |
|Recall          |0.9372401472896423            |


## Conclusions:
- SVMs performed the best among all classifiers with 94% accuracy
- Gaussian outperformed polynomial kernel in almost all iterations
- XGB classifiers were the best among all ensembling methods with 90% accuracy.
- Since genre classes were balanced , the tradeoff between precision and recall was less observed.
- Among all KNN,DT and ensemble classifiers , precision was more than recall
- While in case of LR,SGD,NB,MLP,SVM recall was observed more than precision.
