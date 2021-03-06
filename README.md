
# Robot Inference Project 

## Abstract

#### In this GitHub repository, I show the Blackjack Robot I developed with a Jetson TX2 using AlexNet Model trained on Nvidia Digits. Once the Jetson classified the playing card, though its onboard camera, a decision was outputted to a green (Hit!) and red (Stay) LED. The playing card dataset, used to train the AlexNet Model, was augmented through a python script. The robot has an accuracy of 84% and classified these images in, roughly, 20ms (seen on the Imagenet-Camera). This repository shows and explains the decisions behind: the creation of the playing card dataset, the training of the classification model, and the programming of the decision-making. 

<figure>  <figcaption>Robot Demo #1</figcaption>
<img style="text-align:center;" src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/video/BlackJackRobot_Demo1.gif" description="Demo Of Robot"></img>
</figure>

## History

Blackjack is a card game with the objective to collect cards till one reaches, or gets as close as possible to a total of 21.  It is a game of chance. Unless, one has a unique mathematical skill to count cards. The ability to count cards is difficult to ascertain: counting cards demands quick calculations and keen (inconspicuous) observations. Humans require years of training to develop these skills. Robots, with the right computation power, only need a few hours to count cards. Quick calculations are innate to a computer and- with advances in machine learning and pattern recognition- a robot can develop keen observations skills. My goal with my robot is to develop a robot which will always win at Blackjack and make me millions in Vegas. 

My project of a blackjack robot is not a unique idea. Jason Mecham (Github User S4WRXTTCS) demonstrates in his Jetson-Inference repository <a href=https://github.com/S4WRXTTCS/jetson-inference>(S4WRXTTCS)</a> his Blackjack program. Mecham’s program both detects and classifies playing cards on a table: the detection and classification is completed through the combination of ImageNet and DetectNet. Another unique feature of Mecham’s model is the classification of playing cards’ include the suit of the card. For simplicity my model will only be using ImageNet. The suit of the card (per my understanding) is not important in blackjack, therefore my model will only classify the count of the playing card.  


## Background 

A classification model was trained with the supplied dataset to gain experience with the NVIDIA Digits platform. Googlenet was selected as the Convolutional Neural Network (CNN) architecture based off its accolades- placing 1st in the ImageNet Large Scale Visual Recognition Challenge(ILSVRC)- and its low Top 5 Error Rate of 6.67% <a href=https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5Das>(Das, S 2017)</a>. The parameters (seen below) were established through default parameters from <a href=https://developer.nvidia.com/embedded/twodaystoademo> NVIDIA’s tutorial </a> and experimentation:


**Optimizer:** SGD (Stochastic Gradient Descent)

**Learning Rate:** 0.001

**Epochs:** 15 

**Batch Size:** Default

**Validation Dataset Size:** 10%

**Test Dataset Size:** 5% 


The Googlenet mode did not converge when trained with the Blackjack dataset. Alexnet - the 2012 ILSVRC winner with a Top 5 Error Rate 15.3% -  performed with a high degree of accuracy after 33 epochs. In the previous model, the optimizer selected was SGD. When using Alexnet the Adams optimizer was chosen as it has a lower training and validation error when compared to SGD <a href=https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/>(Shaoanlu 2017)</a>. A lower Learning Rate and more Epochs were selected to ensure a good convergence, and because computation cost was not an issue. Again, the other parameters selected were based off the Digits Tutorial and experimentation. 


**Optimizer:** Adams

**Learning Rate:** 0.00001

**Epochs:** 33

**Batch Size:** Default

**Validation Dataset Size:** 10%

**Test Dataset Size:** 5% 


<figure>  <figcaption>Training for AlexNet</figcaption>
<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/training.PNG" description="Training for AlexNet"></img>
</figure>
<figure>  <figcaption>Training and Learning Rate for AlexNet Image</figcaption>
<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/training%26learning.PNG" description="Training and Learning Rate for AlexNet Image"></img>
</figure>


## Data Aq

Image data of playing cards was collected using the camera on the Jetson TX2. The script ,  took 20 photos for each category of card ( so 20 images of Kings, 20 images of Queens… ). These images were converted to Grayscale: the suit did not matter and therefore color was irrelevant when categorizing. Since 20 images for each category is to small of a dataset for training, the dataset was expanded using a list of data augmentation techniques based off this Medium Post <a href=https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced> (Raj 2018)</a>.

<figure>  <figcaption>Original Image</figcaption>
<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/1.png" Title="Original Image"></img>
</figure>


<figure>  <figcaption>Flipped Image</figcaption>
<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/1_flipped.png" Title="Flipped Image"></img>
</figure>


<figure>  <figcaption>Rotation (From -90° to 90° with a step size of 14)</figcaption>
<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/1_rotated.png" Title="Rotated Image"></img>
</figure>


<figure>  <figcaption>Scale (90%, 75% and 60% scale)</figcaption>
<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/1scaled1.png" Title="Scaled Image"></img>
</figure>


<figure>  <figcaption>Translation (20% Translate in all directions)</figcaption>
<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/1translated1.png" Title="Translated Image"></img>
</figure>


<figure>  <figcaption>Noise (Gaussian & Salt and Pepper)</figcaption>
<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/1_gaus.png" Title="Gaussian Image" ></img>
<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/1_saltpepper.png" Title="Salt&Pepper Image"></img>
</figure>


Not only did this increase the dataset, it highlighted the important characteristics of the cards for the CNN. Here is the augmentation and data acquisition <a href="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/Scripts/camera.py">script</a> developed.

## Results

The GoogleNet model, trained on the supplied dataset, achieved the accuracy and time requirements of 75% and <10ms.
<figure>  <figcaption>GoogleNet Evaluate</figcaption>
<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/GoogleNetTrainSGD.PNG" description="GoogleNet Evaluate"></img>
</figure>

For the The robotic decisions (Hit and Stay) were executed perfectly, see the example below. 

<figure>  <figcaption>Robot Demo #2</figcaption>
<img style="text-align:center;" src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/video/BlackJackRobot_Demo2.gif"></img>
</figure>

Using the Imagenet-camera.cpp script it was seen that the classification time was <=20ms.

<figure>  <figcaption>Inference Results for Blackjack Robot</figcaption>
<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/5inferenceScreenShot.png"></img>
<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/10inferenceScreenShot.png"></img>
</figure>

When testing the the model in the imagenet-console based off 50 new images taken (seperate from the training data) the robot had an accuracy of 84%. Here is the python testing <a href="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/Scripts/pval.py">script</a> created.

<figure>  <figcaption>Blackjack Robot Results</figcaption>
<img src="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/media/images/FinalResults.png" description="Results of Robot-Inference"></img>
</figure>

Testing with the imagenet-camera has some issues, which were not recorded but will be discussed in the next section. 

## Discussion

The accuracy of the model was excellent, and extremely close to that of the first GoogleNet CNN. During the initial testing of Blackjack CNN, a dataset- which was the same size of the final dataset- that had no augmentation was used. The Blackjack CNN trained on this dataset had a high error rate when testing. Based off this, one believes the low error rate is due to the data augmentation. The augmentation highlighted the important patterns for classification to the CNN. It allowed for the position of the playing card to vary without any loss to the accuracy of the model. A wise simulation engineer use to say when reffering to testing and training setups, “Garbage in , garbage out” .

The decisions made by the robot were based off the running total; if < 15 the robot will say hit, if > 15 the robot will stay and if > 21 the game is over. Below is a snipped of the <a href="https://github.com/GlennPatrickMurphy/BlackjackRobot/blob/master/Scripts/imagenet-camera.cpp">C++ script</a> for decision-making. 

```cpp
  if( total < 15 ){
                gpioSetValue(greenLED,on);
                cout << "\n \n Hit Me!Total " << total << "\nPress Enter \n \n";
                cin.ignore();
                gpioSetValue(greenLED,off);
            }


            else if((total>15) && (total<=21)){
                gpioSetValue(redLED,on);
                cout << "\n \n Stay! Total " << total << "\nDeal new cards and Press Enter to Play again \n \n";
                cin.ignore();
                gpioSetValue(redLED,off);
                total=0;
            }


            else if (total>21){
                cout << "\n \n OVER!!! Total " << total << "\nDeal new cards and Press Enter to Play again \n \n ";
                for(int i=0; i<10; i++){
                    usleep(200000);
                    gpioSetValue(redLED,on);
                    gpioSetValue(greenLED,on);
                    usleep(200000);
                    gpioSetValue(greenLED,off);
                    gpioSetValue(redLED,off);
                }
                cin.ignore();
                total=0;
            }

```

When testing with the JetsonTX2 Camera, issues arose due to the shadows on the playing surface. The testing apparatus was moved multiple times (for the demo videos) and the introduced shadows. These shadows on the cameara image seemed to affect the model’s classification.  

## Conclusion

Adding a detectnet would be useful for parallel classifications of cards. Adding a card counting algorithm, to make a robot that wins more games, is important for Vegas. The next iteration will not look to measure the machine based on the accuracy of its classification, but on the percentage of wins vs losses in a game of Blackjack. My fingerless Uncle is looking forward to the next iteration of this robot, he has a lot riding on this. 


## Sources

S4WRXTTCS. (n.d.). S4WRXTTCS/jetson-inference. Retrieved March 11, 2019, from https://github.com/S4WRXTTCS/jetson-inference

Das, S. (2017, November 16). CNN Architectures: LeNet, AlexNet, VGG, GoogLeNet, ResNet and more .... Retrieved March 11, 2019, from https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5

NVIDIA Two Day Tutorial https://developer.nvidia.com/embedded/twodaystoademo

Shaoanlu. (2017, December 29). SGD Adam?? Which One Is The Best Optimizer: Dogs-VS-Cats Toy Experiment. Retrieved March 11, 2019, from https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/

Raj, B. (2018, April 11). Data Augmentation | How to use Deep Learning when you have Limited Data - Part 2. Retrieved March 11, 2019, from https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced

