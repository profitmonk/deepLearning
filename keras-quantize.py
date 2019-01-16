import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import SGD
import readline
import numpy as np
from numpy import newaxis, empty, array
import copy

##Function to assign a modified layerData to a model
def assignLayerDataToModel(model,layerQuantizedData):
    layerIndex = 0;
    for layer in model.layers:
        newLayerIndex = 0
        for newLayer in modelQuantized.layers:
            if newLayerIndex == layerIndex:
                ##newLayer.set_weights(layer.get_weights())
                newLayer.set_weights(layerQuantizedData[layerIndex])
                break
            newLayerIndex += 1
        layerIndex += 1

#Let's get some basic classes going for list manipulation
class listQuantize (object):
    def __init__(self,listIn,listOut,n_bits,multiplierScale):
        self.listIn = listIn
        self.listOut = listOut
        self.n_bits = n_bits
        self.multiplierScale = multiplierScale
    def quantizeListSimple2(self):
        layerIndex = 0
        for layer in self.listIn:
            if (len(layer) > 0):
                i = 0
                layerMax, layerMin = 0, 0
                while i < len(layer):
                    max = np.max(layer[i])
                    min = np.min(layer[i])
                    if layerMax < max:
                        layerMax = max
                    if layerMin > min:
                        layerMin = min
                    i += 1
                i = 0;
                while i < len(layer):
                    resolutionForLayer = (layerMax-layerMin)/2**self.n_bits
                    print("layer Details","layerID:index[",layerIndex,"][",i,"] max,min:",layerMax,layerMin,"resolution:",resolutionForLayer,"\n")
                    self.listOut[layerIndex][i] =  np.round(self.listOut[layerIndex][i]/resolutionForLayer)*resolutionForLayer
                    print("Further","Original number of non zeros:",np.count_nonzero(self.listIn[layerIndex][i]),"non zeros after quantization:",np.count_nonzero(self.listOut[layerIndex][i]),"number of unique values originally:",np.unique(self.listIn[layerIndex][i]).shape[0],"number of unique values after quantization:",np.unique(self.listOut[layerIndex][i]).shape[0],"\n")
                    i += 1
            layerIndex += 1
        return self.listOut
    def quantize_and_replaceSmallValuesWithZeros(self):
        layerIndex = 0
        for layer in self.listIn:
            if (len(layer) > 0):
                i = 0
                layerMax, layerMin = 0, 0
                while i < len(layer):
                    max = np.max(layer[i])
                    min = np.min(layer[i])
                    if layerMax < max:
                        layerMax = max
                    if layerMin > min:
                        layerMin = min
                    i += 1
                i = 0;
                while i < len(layer):
                    resolutionForLayer = (layerMax-layerMin)/2**self.n_bits
                    print("layer Details","layerID:index[",layerIndex,"][",i,"] max,min:",layerMax,layerMin,"resolution:",resolutionForLayer,"\n")
                    self.listOut[layerIndex][i] =  np.round(self.listOut[layerIndex][i]/resolutionForLayer)*resolutionForLayer
                    print("Further","Original number of non zeros:",np.count_nonzero(self.listIn[layerIndex][i]),"non zeros after quantization:",np.count_nonzero(self.listOut[layerIndex][i]),"number of unique values originally:",np.unique(self.listIn[layerIndex][i]).shape[0],"number of unique values after quantization:",np.unique(self.listOut[layerIndex][i]).shape[0],"\n")
                    self.listOut[layerIndex][i][(self.listOut[layerIndex][i] > -self.multiplierScale*resolutionForLayer) & (self.listOut[layerIndex][i] < self.multiplierScale*resolutionForLayer)] = 0
                    print("Further","Original number of non zeros:",np.count_nonzero(self.listIn[layerIndex][i]),"non zeros after zero replacement:",np.count_nonzero(self.listOut[layerIndex][i]),"number of unique values originally:",np.unique(self.listIn[layerIndex][i]).shape[0],"number of unique values after zero replacement:",np.unique(self.listOut[layerIndex][i]).shape[0],"\n")
                    i += 1
            layerIndex += 1
        return self.listOut



#Let us first get some basic data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
##Now we do have the train and test data plugged
##Lets define a cartoon network
##First flatten the MNIST input
##Connect a dense layer --> input shape (784,1), output shape (512,1). Bias is (512,1)
##Add a dropout layers
##Final dense classification layer --> input shape (512,1), output shape (10,1). Bias is (10,1)
model  = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

##Define model optimization parameters
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

##Now train the model. 5 epochs i.e. 5 turns through the full MNIST data.
model.fit(x_train, y_train, epochs=5)

##evaluate and print summary
model.evaluate(x_test, y_test)
model.summary()
model.save_weights('./chkp/chkPnt')

##Now go ahead an quantize all the layers
layerFloatData = []
for layer in model.layers:
    layerFloatData.append(layer.get_weights())
##layerQuantizedData = layerFloatData.copy()
layerQuantizedData_7bits_scale1 = copy.deepcopy(layerFloatData)
layerQuantizedData_5bits_scale1 = copy.deepcopy(layerFloatData)
layerQuantizedData_7bits_scale5 = copy.deepcopy(layerFloatData)
layerQuantizedData_4bits_scale1 = copy.deepcopy(layerFloatData)
layerQuantizedData_3bits_scale1 = copy.deepcopy(layerFloatData)

##Qunatize loop 0-255
## Here is how layer data structure looks like
## isinstance(layer,list) is True..
## The list layer's length is 2 for all dense layers.
## layer[0] is dense weights with shape === input_num_layers * output_num_layers i.e. 784x512 for the first dense layer
## layer[1] is biases array with shape === output_num_layers i.e. 512 for the first dense layer
## layer[0].shape is 784x512, layer[1].shape is 512
## Find max and min across weights and biases and quantize the layer
## QUANTIZATION PROCESS DETAILS
## Configure number of bits for decimal. Value of 6 allows good precision as long as weights are b/w -1.99 and +1.99
## n_bits = 4
## a = np.linspace(-1.99, 1.99, 10000) ##THIS IS BASE ARRAY
## f = (1 << n_bits)
## a_fix = np.round(a*f)*(1.0/f) ## THIS IS QUANTIZED ARRAY
layerIndex, globalMax, globalMin = 0,0,0
for layer in layerFloatData:
    if (len(layer) > 0):
        while i < len(layer):
            max = np.max(layer[i])
            min = np.min(layer[i])
            if globalMax < max:
                globalMax = max
            if globalMin > min:
                globalMin = min
            i += 1
        i = 0;

if (globalMax-globalMin) < 2:
    n_bits = 7
elif (globalMax-globalMin) < 4:
    n_bits = 6
else:
    n_bits = 6

f = (1 << n_bits)

### Let us first quantize layer weights
n_bits = 7
organizedList_7bits = listQuantize(layerFloatData,layerQuantizedData_7bits_scale1,n_bits,1)
layerQuantizedData_7bits = organizedList_7bits.quantizeListSimple2()

n_bits = 5
organizedList_5bits = listQuantize(layerFloatData,layerQuantizedData_5bits_scale1,n_bits,1)
layerQuantizedData_5bits = organizedList_5bits.quantizeListSimple2()

n_bits = 4
organizedList_4bits = listQuantize(layerFloatData,layerQuantizedData_4bits_scale1,n_bits,1)
layerQuantizedData_4bits = organizedList_4bits.quantizeListSimple2()

n_bits = 3
organizedList_3bits = listQuantize(layerFloatData,layerQuantizedData_3bits_scale1,n_bits,1)
layerQuantizedData_3bits = organizedList_3bits.quantizeListSimple2()

## Let's fill even more zeros to assess the impact to results
## similar to layerQuantizedData[1][0][(layerQuantizedData[1][0] <= 5*resolutionForLayer) & (layerQuantizedData[1][0] >= -5*resolutionForLayer)] = 0
n_bits = 7
organizedList_7bits_scale5 = listQuantize(layerFloatData,layerQuantizedData_7bits_scale5,n_bits,5)
layerQuantizedData_7bits_moreZeros = organizedList_7bits_scale5.quantize_and_replaceSmallValuesWithZeros()


## You can use np.count_nonzero(layerQuantizedData - layerFloatData) OR np.count_nonzero(layerQuantizedData > -10*resolutionForLayer) to assess the difference in values and its stats
##


## Let us also quantize the x_test data
n_data_bits = 8
testResolution = (np.max(x_test) - np.min(x_test))/2**n_data_bits
x_test_quantized = np.round(x_test/testResolution)*testResolution

## Now create a quantized model
modelQuantized  = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

modelQuantized.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#modelQuantized.fit(x_train, y_train, epochs=1)

##Now copy the old model weights to new model get_weights and evaluate the model
modelQuantized.set_weights(model.get_weights())
assignLayerDataToModel(modelQuantized,layerQuantizedData_7bits_scale1)
modelQuantized.evaluate(x_test,y_test)
modelQuantized.evaluate(x_test_quantized,y_test)

modelQuantized.set_weights(model.get_weights())
assignLayerDataToModel(modelQuantized,layerQuantizedData_5bits_scale1)
modelQuantized.evaluate(x_test,y_test)
modelQuantized.evaluate(x_test_quantized,y_test)

modelQuantized.set_weights(model.get_weights())
assignLayerDataToModel(modelQuantized,layerQuantizedData_7bits_scale5)
modelQuantized.evaluate(x_test,y_test)
modelQuantized.evaluate(x_test_quantized,y_test)

modelQuantized.set_weights(model.get_weights())
assignLayerDataToModel(modelQuantized,layerQuantizedData_4bits_scale1)
modelQuantized.evaluate(x_test,y_test)
modelQuantized.evaluate(x_test_quantized,y_test)

modelQuantized.set_weights(model.get_weights())
assignLayerDataToModel(modelQuantized,layerQuantizedData_3bits_scale1)
modelQuantized.evaluate(x_test,y_test)
modelQuantized.evaluate(x_test_quantized,y_test)
##You can copy the weights from old model to the new model, layer by layer as well, if needed but using model.get_weights is way easier
layerIndex = 0;
for layer in model.layers:
    newLayerIndex = 0
    for newLayer in modelQuantized.layers:
        if newLayerIndex == layerIndex:
            ##newLayer.set_weights(layer.get_weights())
            newLayer.set_weights(layerQuantizedData[layerIndex])
            break
        newLayerIndex += 1
    layerIndex += 1
