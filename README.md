# Music-Genre-Classification

## Abstract
We continually strive to improve our music experience by exploring different resources. We started our music journey with radio programs and television channels and gradually used infamous websites to offline access music. Now, with the streaming platforms, we have achieved the “epitome” of satisfaction, yet, sometimes, whenever we try to explore new music, the streaming platforms cannot provide an entirely satisfactory experience to users.

We have a quandary approach towards segregating our music based on genres and style. Our objective then became to resolve this by applying machine learning techniques to categorize or cluster the music into genres and provide a mechanism for the user to explore new music based on it. By referring to many models already existing on the internet and other resources, we aim to incorporate a comprehensive range of music tastes. We do not seek to solve the issue but take a step towards an even better experience.

## Introduction
We aim to address music classification into specific musical genres through our model and further extend this into practical applications. For this, we have done a thorough research and preprocessing by displaying the human as well as computer-interpreted graphs and labels. This is accompanied by clustering and an overall idea of the music we are dealing with in our dataset. We further propose Multi Classification models for the provided music dataset to categorize it into specific music genres. We have analyzed and tested 7-8 machine learning models and thoroughly experimented to achieve the best achievable performance, including hyperparameter tuning and a performance report. We aim to pick the best model and use it for our further research and implementation.

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

__Standardization__: The aim of this step is to standardize the range of the continuous initial variables so that each one of them contributes equally to the analysis.

More specifically, the reason why it is critical to perform standardization prior to PCA, is that the latter is quite sensitive regarding the variances of the initial variables. That is, if there are large differences between the ranges of initial variables, those variables with larger ranges will dominate over those with small ranges (For example, a variable that ranges between 0 and 100 will dominate over a variable that ranges between 0 and 1), which will lead to biased results. So, transforming the data to comparable scales can prevent this problem.
 
 ## Methodology
 Feature extraction -> correlation matrix -> PCA
 - With 30 secs sample
 ![](Features_extracted_plots/Corr_Heatmap30_better.png)
 ![](https://github.com/ashcode028/Music-Genre-Classification/blob/ef966977e66f7f62a116ffb6c3b2a2a4f3f3f8cc/Features_extracted_plots/PCA30_std.png)
 - With 3 secs sample
![](Features_extracted_plots/Corr_Heatmap3.png)
![](https://github.com/ashcode028/Music-Genre-Classification/blob/ef966977e66f7f62a116ffb6c3b2a2a4f3f3f8cc/Features_extracted_plots/PCA_std.png)
  
