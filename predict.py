import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import dataset
# from dataset import *
from numpy import genfromtxt
# labels = genfromtxt('labels.txt', delimiter='\n')
# print(labels)
labels = []
with open('labels.txt', 'r') as f:
    for line in f:
        labels.append(line.strip())

print(labels)
# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path)
image_path=sys.argv[1] 
filename = dir_path +'/' +image_path
# print(filename)
# dir_path = os.path.dirname(os.path.realpath(''))

# # dir_path = '/home/rounak/Documents/trial/'
# # image_path=sys.argv[1] 
# # dir_path = os.path.abspath(os.path.dirname(sys.argv[1]))
# image_path = 'trial/papaya.jpg'
# filename = dir_path + '/' +image_path
# print(filename)

image_size=160
num_channels=3
images = []
# Reading the image using OpenCV
image = cv2.imread(filename)
# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0) 
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size,image_size,num_channels)

## Let us restore the saved model 
sess = tf.Session()

# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('/home/rounak/Documents/trial/models/fruits-model.meta')
# saver = tf.train.import_meta_graph('/home/corse/st119979/Documents/project/new_models/fruits-model.meta')

# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('/home/rounak/Documents/trial/models/'))
# saver.restore(sess, tf.train.latest_checkpoint('/home/corse/st119979/Documents/project/new_models/'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")
# dataset.load_train()
## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, len(os.listdir('training_data')))) 


### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
# result is of this format [probabiliy_of_mango probability_of_apple]
# abc=sess.run(y_pred)
# print("\n",abc)
print("\n")

resultArray = np.argsort(result)
r=np.amax(result)
print( result)
# print("\n")
# shape = resultArray.shape
print(resultArray[0,13])
# print("\n")
print(labels[resultArray[0,-1]])
print(r)
# print(index)
# cv2.putText(labels[resultArray[0,13]])

cv2.putText(image, str(labels[resultArray[0,-1]])+ '=' + str(r), (5,10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0),1)
# cv2.putText(image,labels[resultArray[0,-1]]), (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

cv2.namedWindow('image',0)
cv2.imshow("image",image)
# if cv2.waitKey(1) & 0xFF == ord('q'):
    # break
cv2.waitKey(0)

cv2.destroyAllWindows()

def plot_confusion_matrix(y_true,y_pred):
    cm_array = confusion_matrix(y_true,y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    plt.imshow(cm_array[:-1,:-1], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks,pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size

plot_confusion_matrix(y_true,y_pred)
# cv2.imshow(filename)
# cv2.waitkey()

