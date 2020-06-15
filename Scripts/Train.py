import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, \
                                    ReLU, BatchNormalization, \
                                    MaxPool2D, \
                                    Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# tf.debugging.set_log_device_placement(True)

# # Create some tensors
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)

# print(c)


batchS = 32
dataResize = (48, 48)
eps = 50
dataDir = "/home/devin/MyGit/TfLab/FrontalProfileFace/Data/TrainData"

class SiameseNet(Model):
    """ Implementation for Siamese networks. """

    def __init__(self):
        super(SiameseNet, self).__init__()

        # CNN-encoder
        self.encoder = tf.keras.Sequential([
            Conv2D(filters=8, kernel_size=3, padding='valid'),
            ReLU(),
            MaxPool2D((2, 2)),

            Conv2D(filters=12, kernel_size=3, padding='valid'),
            ReLU(),
            MaxPool2D((2, 2)),

            Conv2D(filters=16, kernel_size=3, padding='valid'),
            ReLU(),
            MaxPool2D((2, 2)),

            Conv2D(filters=16, kernel_size=2, padding='valid'),
            ReLU(),

            Flatten(),
            Dense(1, activation='sigmoid')]
        )

    @tf.function
    def call(self, x):
        z = self.encoder(x)
        return z

dataGen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.15,
        horizontal_flip=True)

trainData = dataGen.flow_from_directory(batch_size=batchS, \
                                        directory=dataDir, \
                                        shuffle=True, \
                                        target_size=dataResize, \
                                        class_mode='binary', \
                                        interpolation='bilinear', \
                                        subset='training')

valData = dataGen.flow_from_directory(batch_size=batchS, \
                                        directory=dataDir, \
                                        shuffle=False, \
                                        target_size=dataResize, \
                                        class_mode='binary', \
                                        interpolation='bilinear', \
                                        subset='validation')


# sample_training_images, _ = next(trainData)

print()
model = SiameseNet()
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
history = model.fit(
    trainData, \
    epochs=eps, \
    validation_data=valData
)
model.save("../tmp/20200060910")