import os
import subprocess
import tempfile
import random
import time


def runtest(string):
    out = subprocess.Popen(['./imagenet-console', string, 'output_0.jpg ',
                            '--prototxt=networks/blackjack_model/blackjack_model/deploy.prototxt',
                            '--model=networks/blackjack_model/blackjack_model/snapshot_iter_3380.caffemodel',
                            '--labels=networks/blackjack_model/blackjack_model/labels.txt',
                            '--input_blob=data',
                            '--output_blob=softmax'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return out.communicate()


def gotobin(string):
    return os.system(string)


def main():

    arr = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K']

    correct=float(0.00)

    inncorrect=float(0.00)

    final=0.000

    t=0

    for x in range(50):

	rand_class=arr[random.randint(1, 13)-1]

        filepath = "Images//"+rand_class+"//"+str(random.randint(1, 20))+".png"
	
	tic=time.time()

        output = runtest(filepath)

	toc=time.time()-tic
	
	str1=output[0]

	item=str1.find("->")

	resp=str1[item+23:item+25]
	
	if resp[0]=="(":
		resp=resp[1]
	else:
		resp=resp[0]
	
	t=toc+t
	
	print("-------------\nCard: "+rand_class+"\nGuess: "+resp)

	if x>0:
		t=t/2
		print(t)

	if resp==rand_class:
		correct+=1.00
		print("\nAmount correct "+ str(correct)+"\n-------------")

	else:
		inncorrect+=1.00
		print("\nAmount wrong"+str(inncorrect)+"\n-------------")
    print("\n-------------\nAmount correct "+str(correct))
    print("\nAmount Total "+str(inncorrect+correct))
    final=(float(correct)/(float(correct)+float(inncorrect)))*100
    print("Accuracy Percentage")
    print(final)
    print("Average Time in seconds")	    
    print(t)

if __name__ == '__main__':
    main()

