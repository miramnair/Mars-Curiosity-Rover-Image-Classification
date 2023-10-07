# Mars-Curiosity-Rover-Image-Classification
A Comparative Evaluation of Transfer Learning, CBAM, and Autoencoder Approaches

**CNN with Convolutional Block Attention Module (CBAM)**

**Core concept**: This model utilizes “attention” mechanism. Attention mechanism improves the predictive accuracy and computational efficiency of deep learning models by selectively concentrating on significant input elements. 
   When humans direct their attention towards a specific object within their visual field, the attention mechanism inside their visual perception system utilizes a complex arrangement of filters to generate a blurring phenomenon. This blurring effect serves to prioritize the object of interest, while causing the surrounding elements to diminish in prominence (Misra, 2020). “Attention” in machine learning emulates this human visual perception.

**Input Image size** :  224 pixels in width, 224 pixels in height, and three color channels (RGB) for each pixel (224x224x3).

**Architecture** : In this model, a CNN with five convolutional layers was developed. It was restricted to five due to the system's computational limitations. The output of these layers is then transmitted to a CBAM attention module, which enables it to learn and focus more on essential information as opposed to extraneous secondary information. The output of CBAM is then passed to three fully connected layers (dense layers) with ReLU activation. Finally, softmax activation is used in the output layer to perform multi class classification. Additionally, the following regularization techniques have been applied to improve the performance of the model :
•	Batch normalization, to minimize the covariance shift.
•	Random dropout of 0.4. 
•	L2 regularization, to minimize loss function by adding penalties to larger weights.

To compare the performance, accuracy was calculated with and without the CBAM module in the CNN.


![image](https://github.com/miramnair/Mars-Curiosity-Rover-Image-Classification/assets/128325004/414aa4b2-6b6b-46f8-a384-232389c0c65b)

**Autoencoder approach**

**Core Concept** : This model utilizes semi-supervised learning.

**Input Image size** :  124 pixels in width, 124 pixels in height, and three color channels (RGB) for each pixel (124x124x3).

**Architecture** : It employs a two-stage training procedure: 

•	In the first phase, begin by removing the labels from the training dataset. Then a Convolutional autoencoder (CAE) is trained using unlabeled images (unsupervised learning) for image feature learning. Convolutional autoencoder (CAEs) are unsupervised models comprised of convolutional layers that are able to generate compressed image representations (Martinez & Stiefelhagen, 2018). In general, CAEs are primarily employed for dimensionality reduction, eliminating noise while maintaining all useful information, and feature engineering. The CAE developed in this study has the following structure : 

**Encoder**: It consists of four convolution blocks, and each of these blocks contains two convolution layers, followed by a batch normalization layer. The main function of an encoder is to extract the most relevant features. The extracted features are compressed into the smallest vector representation (31x31x256 in Fig. 22). This is done to make it harder for the decoder to interpret the features and to force it to learn more complex mappings. The trained weights are saved for later use.
**Decoder**: It is composed of three convolution blocks with two convolution layers each, followed by a batch normalization layer. In addition, a final convolutional layer with sigmoid activation is added at the end. Upsampling is used to increase the dimensions of the input from the previous layers. Here, the decoder tries to make sense of the compressed features from the encoder, in order to reconstruct the original image.


![image](https://github.com/miramnair/Mars-Curiosity-Rover-Image-Classification/assets/128325004/163259ec-c376-426f-84ac-28bc6fff9b44)

•	In the second phase, the weights trained by the CAE are used to retrain the labeled images (supervised learning) to perform image classification using a CNN. The CNN has the same structure as the encoder in CAE. First, load the weights of the autoencoder (from the first phase) to the CNN and then freeze the layers of the CNN. This means the weights of the CNN layer cannot be modified further. By doing so, one forces the CNN to perform image classification based on the weights learned during unsupervised learning using CAE. Finally, add two fully connected layers to the model, one with ReLu activation and the last using softmax activation in order to perform multiclass classification of images


![image](https://github.com/miramnair/Mars-Curiosity-Rover-Image-Classification/assets/128325004/b028a6bd-79d0-45bb-9ba8-99363d9919c1)


**Transfer learning using Xception**

**Core Concept** : This model utilizes transfer learning, where the weights learned by a previously trained model on the ImageNet dataset are re-used. The selection of Xception was based on its superior accuracy in comparison to other prevalent models.

**Input Image size** :  224 pixels in width, 224 pixels in height, and three color channels (RGB) for each pixel (224x224x3).

Architecture The Xception architecture is a convolutional neural network that utilizes depthwise separated convolutions. It has 36 convolutional layers organized into 14 modules, all but the first and last of which have linear residual connections surrounding them. (Chollet, 2017).


**Implementation** :  

1)	Import the Xception model from keras library.
2)	The model expects an input shape of 299x299x3 (height, width, channel). Update this to accommodate a 224x224x3 input image.
3)	Load the pretrained weights trained on the ImageNet dataset.
4)	Freeze the weights of the layer to prevent the training procedure from updating them and forcing it to use the weights learned from ImageNet.
5)	Add two fully connected layers, one with ReLu activation, and the second one with softmax activation to perform multi class classification.


![image](https://github.com/miramnair/Mars-Curiosity-Rover-Image-Classification/assets/128325004/5a874fb5-d8e0-4d82-b3f8-436999a4e1fa)


**Testing**

   For training and evaluating the models, the data has been divided into train, test and validation datasets (Table 3).

•	The models are trained on training datasets so that they can discover hidden features and patterns within the data. In each epoch, the neural network continues to receive the same training data, and the model keeps learning the properties of the data. 

•	The validation set is used to check how well the models are doing during training. This validation procedure provides information that enables us to adjust the hyperparameters and configurations of the model appropriately. To avoid overfitting, the model is trained on the training set and tested on the validation set after each epoch. 

•	A test dataset, which is an independent set of data used to test the model once training is complete, gives an impartial final metric of model accuracy.

   Image size, compiler settings and hyperparameter tuning are other crucial variables that can impact the performance of a test. It is an iterative process requiring experimentation, monitoring of results, and comparison to determine the optimal parameters for optimal performance.

   ![image](https://github.com/miramnair/Mars-Curiosity-Rover-Image-Classification/assets/128325004/2fdce12d-b929-419d-abce-5525e2ea5a02)

