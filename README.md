
## Support Vector Machine based Vehicle Make and Model Recognition  
# Table of Contents:  
1. Introduction  
2. Problem set up.  
3. Loading Dataset.  
4. How to execute the code.  
5. Execution of the code.  
# Introduction  
In this project, we are trying to implement vehicle make and model classification using Scale Invariant Feature Transform(SIFT) for feature extraction from images and Bag Of Words(BoW) feature representation algorithm that creates the feature vector that is fed to the Support Vector Machine(SVM). The dataset used is a subset of the publicly available National Taiwan Ocean University-Vehicle Make and Model Recognition(NTOU-VMMR) dataset.  
# Problem set up  
For classifying images, we need data related to that image. The SIFT algorithm converts the information in the image, which is stored as pixels, to feature vectors. A given image can have multiple keypoints, points on the image that give information. Each of these keypoints have a fixed number of descriptors (128). So, the output of the SIFT algorithm would be n x d, where n is the number of key points and d is the descriptors.  
To the SVM this is just a collection of numbers. So, before feeding to the SVM classifier, the extracted features are grouped together. This enables the possibility of distinct classifications when given to SVM. The grouping together, or clustering, is done by k means clustering. The more the number of clusters, the better the accuracy of the overall classifier will be. This process is called Bag-Of-Words(BoW) feature representation. Using these, clusters, it is possible to develop a “dictionary” of words, which are essentially a group of keypoints having features that represent a particular object together. In this case, it would be the make and model of the car.  
Now, we feed this output to a Support Vector Machine(SVM). Using a soft SVM, a classifiers that allows for a margin of error, is usually followed. The SVM classifies by creating a hyperplane that separates the data into classes.  
We have implemented all of the above in the codes provided.  
# Loading Dataset  
The data set we have taken consist of 860+ images and 12 different vehicles. Please have a look at the data set provided. The folder, containing the images, have names that are the
make and model of the vehicles. For convenience, we have divided the images into 2 sets: training and testing data.  
We have developed the code in Google Colab. The data was initially uploaded to the corresponding google drive.  

![](https://github.com/ksubra01/Make_and_model_Recog/blob/main/load_data.png)
  
This line of code will load your drive to Google Colab. Please make sure to give necessary permissions when prompted.  
The label_image.py file will create read the image and assign the name of the make and model by using the folder in which the images is present. For each class of vehicle, a number is assigned. The first vehicle( in our case, Audi A5) will get mapped to number 0. The next class, will get mapped to 1 and so on. The function label_image.py returns the list of images (each image as a 3d matrix), list of numerical mappings of each class of vehicle and a dictionary in which the keys are the vehicle make and model and the corresponding values are the number assigned to that class. 
![](https://github.com/ksubra01/Make_and_model_Recog/blob/main/pic2.png)
The data is loaded and the required input matrix and labels are ready to be used.  

Data used for training and testing:  
![](https://github.com/ksubra01/Make_and_model_Recog/blob/main/image.png)
# How to execute the code
• If using Google Colab, make sure to upload the data set to your drive before execution.  
• Load Google drive to Google colab following the steps given above.  
• Copy and paste the path of the folder containing the data in your google drive at the place indicated in code.  
• Run the functions first.  
• Run the main file to get details on the train and test accuracy and the corresponding plot.  
For convenience, we have included the entire python notebook. This notebook need only be executed after making suitable changes to the path.  
# Execution of the code  
• After loading the data, each image matrix is passed to SIFT_feature_extraction() function. This function returns a matrix for each image. The size of the matrix will be n x d, where n is the number of key points in the image and d is the number of descriptors (=128). The number of key points may be different for each image.  
![](https://github.com/ksubra01/Make_and_model_Recog/blob/main/pic3.png)
• The Clustering and the feature_vector_creation functions are part of the Bag of Words implementation. The Clustering() function performs k means clustering on the extracted SIFT descriptors and creates what’s known as a “visual dictionary” using the pickle library. The feature_vector_creation() function takes in the SIFT image descriptors, the number of clusters(vocabulary size) and clustered model. The output of this function is the required feature vector.  
![](https://github.com/ksubra01/Make_and_model_Recog/blob/main/pic4.png)
• This feature vector for each image is of dimensions vocabulary size x 1. This feature vector is split into train and test and given as input to the SVM with different regularization constants and various kernel configurations in the main code for SVM. In the main part of the code, the above mentioned functions are called and the resulting feature vector is used to train and test SVM.  
