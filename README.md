# ICP_10

Complete the following:

Name (Last_First): Blundell Jimmy

Email : jdbkv5@umsystem.edu

Summary of ICP step-wise with adaquate screenshots below:

In this lesson, we discussed ANNs and Recurrent Neural Networks.
Specifically in the ICP, we were given code that contained multiple erros that prevented it from running suscessfully.
There erros I noted were: the lack of an appropriate input_dim parameter, an incorrect activation function (changed from sigmoid to softmax), and some missing import statements to work with additional libraries needed to complete the assignments (I specifically imported pad_sequences).

Fixing these errors allowed my code to run. For starters, here is my code of the original sentiment analysis after the fixes:<br> 
![](https://github.com/JimmyBlundell/python-ICP10/blob/master/screenshots/sentiment-code.png)<br>
Running this code gave the following output, along with the respective plots as we were asked to code:<br>
![](https://github.com/JimmyBlundell/python-ICP10/blob/master/screenshots/sentiment-epochs.png)<br>
![](https://github.com/JimmyBlundell/python-ICP10/blob/master/screenshots/sentiment-accuracy.png)<br>
![](https://github.com/JimmyBlundell/python-ICP10/blob/master/screenshots/sentiment-loss.png)<br>
The runtime for this model was decent, though the accuracy never really got over ~ 50%.


Next, we added an embedded layer to the same model. The code is very similar, the only difference being the added embedded layer as you see:<br>
![](https://github.com/JimmyBlundell/python-ICP10/blob/master/screenshots/embedding-code.png)<br>
Running the code this time gave the following output along with the respective plots:<br>
![](https://github.com/JimmyBlundell/python-ICP10/blob/master/screenshots/embedding-epochs.png)
![](https://github.com/JimmyBlundell/python-ICP10/blob/master/screenshots/embedding-accuracy.png)
![](https://github.com/JimmyBlundell/python-ICP10/blob/master/screenshots/embedding-loss.png)<br>
There were slightly more epochs needed to bring the accuracy to a similar level here, but the loss dropped much faster. I should also note that the runtime when adding the Embedding layer was drastically increased. Runtime without the layer was roughly 2-3 minutes, while runtime when adding the layer took upwards of 10 minutes. Since there was no noticeable increase in the accuracy, I'm not sure that adding the layer is a viable method for this project.

Lastly, we were asked to run the same model as our first one, but on a different dataset: the 20newsgroups. The code is as follows:<br>
![](https://github.com/JimmyBlundell/python-ICP10/blob/master/screenshots/20newsgroups-code.png)<br>
Once again, running this code gave the following output and plots:
![](https://github.com/JimmyBlundell/python-ICP10/blob/master/screenshots/20newsgroups-epochs.png)
![](https://github.com/JimmyBlundell/python-ICP10/blob/master/screenshots/20newsgroups-accuracy.png)
![](https://github.com/JimmyBlundell/python-ICP10/blob/master/screenshots/20newsgroups-loss.png)<br>
While the loss was not greatly affected, the accuracy for this dataset was much higher than the previous one. Runtime remained roughly the same.
