import numpy as np
from tensorflow import keras
from conv import Conv3x3
from pool import Pool2
from softmax import softmax

# Load MNIST from Keras
(trainsamples, trainlabels), (testsamples, testlabels) = keras.datasets.mnist.load_data()

conv=Conv3x3(8)  # 28x28x1 -> 26x26x8
pool=Pool2()     # 26x26x8 -> 13x13x8
softmax_layer=softmax(13*13*8,10) # 13x13x8 -> 10

def forward(image,label):
    # Forward pass
    out=conv.forward((image/255)-0.5)
    out=pool.forward(out)
    out=softmax_layer.forward(out)

    # Calculate loss and accuracy
    loss=-np.log(out[label])
    if(np.argmax(out)==label):
        acc=1
    else:
        acc=0

    return out,loss,acc

def train(image,label,learning_rate=0.005):
    # Forward
    out,loss,acc=forward(image,label)

    # Initial gradient
    gradient=np.zeros(10)
    gradient[label]=-1/out[label]

    # Backward
    gradient=softmax_layer.backprop(gradient,learning_rate)
    gradient=pool.backprop(gradient)
    conv.backprop(gradient,learning_rate)

    return loss,acc

print('MNIST CNN initialized!')
for epoch in range(3):
    print('--- Epoch %d ---'%(epoch+1))

    # Shuffle training data
    permutation = np.random.permutation(len(trainsamples))
    trainsamples = trainsamples[permutation]
    trainlabels = trainlabels[permutation]

    loss=0
    num_correct=0
    for i, (im, label) in enumerate(zip(trainsamples, trainlabels)):
        if i % 1000 == 999:
            print(
                '[Step %d] Past 1000 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 1000, num_correct / 10)
            )
            loss = 0
            num_correct = 0

        l, acc = train(im, label)
        loss += l
        num_correct += acc
print('Training done!')