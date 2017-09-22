# Behavioral Cloning Project

This repository presents the code to train a deep neural network to clone driving behavior, by mappming raw pixels from a single front-facing camera directly to steering commands.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation. 

[//]: # (Image References)

[image1]: ./images/center.jpg "Center Image"
[image2]: ./images/side_cameras.png "Multiple Cameras"
[image3]: ./images/left.jpg "Left Image"
[image4]: ./images/right.jpg "Right Image"
[image5]: ./images/left_cw.jpg "Left Image Clockwise"
[image6]: ./images/center_cw.jpg "Center Image Clockwise"
[image7]: ./images/right_cw.jpg "Right Image Clockwise"
[image8]: ./images/architecture.png "CNN Architecture"

---

## Creation of the Training/Validation Set

To capture good driving behavior, I first recorded four laps on track one in a counter-clockwise direction, focusing on center lane driving as well as smooth streering commends around curves. Here is an example image from the center camera:

![alt text][image1]

To train the model to be able to recover from being off-center, I used the side camera images, which are associated with adjusted steering angles as illstrated below:

![alt text][image2]

Here are example images from the left and right cameras:

![alt text][image3]
![alt text][image4]

Then I repeated this process on track one in clockwise direction, in order to aviod a left turn bias. Here are example images from the left, center and right cameras in clockwise direction:

![alt text][image5]
![alt text][image6]
![alt text][image7]

To augment the dataset, I also flipped images and angles thinking that this would help the model generalize better. After the collection process, I had 58836 data points. To work with such large amounts of data in a momory-efficient way, I used a Python generator to pull pieces of the data and process them on the fly. The code to record the path of images and steer angles is shown below:

```python
path = 'data_track1'
samples = []
with open(path + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # Record the path of center camera images and steer angles
        center_path = path + '/IMG/' + line[0].split('\\')[-1]
        center_angle = float(line[3])
        # To flip images within the generator function, add flag_flip as the third entry
        # flag_flip=1 for original image, flag_flip=0 for flipped image
        samples.append((center_path, center_angle, 1))

        # Create adjusted steering measurements for the side camera images
        correction = 0.1  # this is a parameter to tune
        left_path = path + '/IMG/' + line[1].split('\\')[-1]
        left_angle = center_angle + correction
        samples.append((left_path, left_angle, 1))
        right_path = path + '/IMG/' + line[2].split('\\')[-1]
        right_angle = center_angle - correction
        samples.append((right_path, right_angle, 1))

        # Flip images horizontally for data augmentation
        samples.append((center_path, -center_angle, -1))
        samples.append((left_path, -left_angle, -1))
        samples.append((right_path, -right_angle, -1))
```

The generator function is defined as follows:

```python
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = mpimg.imread(batch_sample[0])
                angle = float(batch_sample[1])
                if batch_sample[2] > 0:
                    images.append(image)                # original image
                else:
                    images.append(cv2.flip(image, 1))   # flipped image
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
```

Then, the dataset was shuffeled and splitted into a training and validation set, such that 20% was for validation:

```python
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
```

Finally, I preprocessed this data by cropping images and normalization, using a Cropping2D layer and lambda layer in Keras:

```python
model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x/255.0) - 0.5))
```

## Design and Test a Model Architecture

My final model architecture consisted of 8 layers, including a cropping/normalization layer, 5 convolutional layers, and 2 fully connected layers. Here is a visualization of the architecture:

![alt text][image8]

The first 3 convolutional layers consisted of 16, 24, 32 (respectively) 5x5 filters, followed by 2x2 max pooling. The last two convolutional layers used 48 3x3 filters and 64 2x2 filters, respectively. The output of the fifth convolutional layer was flatten and fed to 2 fully connected layers, with were composed of 128 and 16 neurons, respectively. In addition, the model includes RELU layers to introduce nonlinearity, and dropout layers to prevent overfitting. The code to defind the network architecture in Keras is given below:

```python
# Define network architecture using Keras
model = Sequential()
# Layer 1: Cropping Images
model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))
# Normalization
model.add(Lambda(lambda x: (x/255.0) - 0.5))

# Layer 2: 5x5 Convolutional. Input = 80x320x3. Output = 76x316x16.
model.add(Conv2D(16, 5, 5, activation='relu'))
# 2x2 Pooling. Output = 38x158x16.
model.add(MaxPooling2D())

# Layer 3: 5x5 Convolutional. Output = 34x154x24.
model.add(Conv2D(24, 5, 5, activation='relu'))
# 2x2 Pooling. Output = 17x77x24.
model.add(MaxPooling2D())

# Layer 4: 5x5 Convolutional. Output = 13x73x32.
model.add(Conv2D(32, 5, 5, activation='relu'))
# 2x2 Pooling. Output = 6x36x32.
model.add(MaxPooling2D())

# Layer 5: 3x3 Convolutional. Output = 4x34x48.
model.add(Conv2D(48, 3, 3, activation='relu'))
# 2x2 Pooling. Output = 2x17x48.
model.add(MaxPooling2D())

# Layer 6: 2x2 Convolutional. Output = 1x16x64.
model.add(Conv2D(64, 2, 2, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))

# Layer 7: Fully Connected. Input = 1024. Output = 128.
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Layer 8: Fully Connected. Output = 16.
model.add(Dense(16, activation='relu'))

# Output = steer angle.
model.add(Dense(1))
```

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
