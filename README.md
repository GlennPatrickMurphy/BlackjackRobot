
# Robot Inference Project 

## Abstract

#### In this GitHub repository, I show the Blackjack Robot I developed with a Jetson TX2 using AlexNet Model trained on Nvidia Digits. Once the Jetson classified the playing card, though its onboard camera, a decision was outputted to a green (Hit!) and red light (Stay). The playing card dataset, used to train the AlexNet Model, was augmented through a python script. The robot has an accuracy of 84% and classified these images with an average time of 3s. This repository shows and explains the decisions behind: the creation of the playing card dataset, the training of the classification model, and the programming of the decision-making capabilities. 


<img style="text-align:center;" src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/video/BlackJackRobot_Demo1.gif"></img>

## History

Blackjack is a card game with the objective to collect cards till one reaches, or gets as close as possible to a total of 21.  It is a game of chance. Unless, one has a unique mathematical skill to count cards. The ability to count cards is difficult to ascertain: counting cards demands quick calculations and keen, inconspicuous, observations. Humans require years of training to develop these skills. Robots,with the right computation power, only need a few hours to count cards. Quick calculations are innate to a computer and- with advances in machine learning and pattern recognition- a robot can develop keen observations skills. My goal with my robot is to develop a robot which will always win at Blackjack and make me millions in Vegas. 

My project of a blackjack robot is not a unique idea. Jason Mecham (Github User S4WRXTTCS) demonstrates in his Jetson-Inference repository <a href=https://github.com/S4WRXTTCS/jetson-inference>(S4WRXTTCS)</a> his Blackjack program. Mecham’s program both detects and classifies playing cards on a table: the detection and classification is completed through the combination of ImageNet and DetectNet. Another unique feature of Mecham’s model is classification of playing cards’ include the suit of the card. For simplicity my model will only be using ImageNet. The suit of the card, per my understanding, is not important in blackjack, therefore my model will only classify the count of the playing card.  


## Background 

A classification model was trained with the supplied dataset to gain experience with the NVIDIA Digits platform. Googlenet was selected as the Convolutional Neural Network (CNN) architecture based off its accolades- placing 1st in the ImageNet Large Scale Visual Recognition Challenge(ILSVRC)- and its low Top 5 Error Rate of 6.67% <a href=https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5Das>(Das, S 2017)</a>. The parameters (seen below) were established through default parameters from <a href=https://developer.nvidia.com/embedded/twodaystoademo> NVIDIA’s tutorial </a> and experimentation:


Optimizer: SGD (Stochastic Gradient Descent)

Learning Rate: 0.001

Epochs: 15 

Batch Size: Default

Validation Dataset Size: 10%

Test Dataset Size: 5% 


<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/10inferenceScreenShot.png"></img>


The Googlenet mode did not converge when trained with the Blackjack dataset. Alexnet - the 2012 ILSVRC winner with a Top 5 Error Rate 15.3% -  performed with a high degree of accuracy after 33 epochs. In the previous model, the optimizer selected was SGD. When using Alexnet the Adams optimizer was chosen as it has a lower training and validation error when compared to SGD <a href=https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/>(Shaoanlu 2017)</a>. A lower Learning Rate and more Epochs were selected to ensure a good convergence, and because computation cost was not an issue. Again, the other parameters selected were based off the Digits Tutorial and experimentation. 


Optimizer: Adams

Learning Rate: 0.00001

Epochs: 33

Batch Size: Default

Validation Dataset Size: 10%

Test Dataset Size: 5% 

<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/training.PNG"></img>
<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/training%26learning.PNG"></img>



## Data Aq

Image data of playing cards was collected using the camera on the Jetson TX2. The script ,  took 20 photos for each category of card ( so 20 images of Kings, 20 images of Queens… ). These images were converted to Grayscale, the suit did not matter and therefore color was irrelevant when categorizing. Since 20 images for each category is to little, the dataset was expanded using a list of data augmentation techniques based off this Medium Post <a href=https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced> (Raj 2018)</a>.

-Flip
-Rotation
-Scale
-Crop
-Translation
-Noise (Gaussian & Salt and Pepper)
-Translation
 
Not only did this increase the dataset, it highlighted the patterns on the cards that CNN would use to categories.   

## Results

<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/GoogleNetTrainSGD.PNG"></img>

The robotic decisions (Hit and Stay) were executed perfectly, see the example below. 

<img style="text-align:center;" src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/video/BlackJackRobot_Demo2.gif"></img>

<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/5inferenceScreenShot.png"></img>
<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/10inferenceScreenShot.png"></img>

<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/FinalResults.png"></img>

When testing the the model in the imagenet-console based off 50 new images taken (seperate from the training data) the robot had an accuracy of 

Testing with the imagenet-camera has some issues, which were not recorded but will be discussed in the next section. 

## Discussion

The accuracy of the model was excellent, extremely close to that of the first GoogleNet CNN. I think these results were based off the data augmentation. The augmentation highlighted the important patterns for classification to the CNN. It allowed for the position of the playing card to vary without any loss to the accuracy of the model. “Garbage in , garbage out” . When testing with the JetsonTX2 Camera, issues arose due to the shadows on the playing surface. The testing apparatus was moved multiple times, for video of the testing, and the introduced shadows on the image seemed to affect the model’s classification.  

## Conclusion

Adding a detectnet would be useful for multiple classifications of cards. Adding a card counting algorithm, to make a robot that wins more games is important for Vegas. The next iteration will not look to measure the machine based on the accuracy of its classification but on the percentage of wins vs losses in a game of Blackjack. My fingerless Uncle is looking forward to the next iteration of this robot. 



## Sources

S4WRXTTCS. (n.d.). S4WRXTTCS/jetson-inference. Retrieved March 11, 2019, from https://github.com/S4WRXTTCS/jetson-inference

Das, S. (2017, November 16). CNN Architectures: LeNet, AlexNet, VGG, GoogLeNet, ResNet and more .... Retrieved March 11, 2019, from https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5

NVIDIA Two Day Tutorial https://developer.nvidia.com/embedded/twodaystoademo

Shaoanlu. (2017, December 29). SGD Adam?? Which One Is The Best Optimizer: Dogs-VS-Cats Toy Experiment. Retrieved March 11, 2019, from https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/

Raj, B. (2018, April 11). Data Augmentation | How to use Deep Learning when you have Limited Data - Part 2. Retrieved March 11, 2019, from https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced

