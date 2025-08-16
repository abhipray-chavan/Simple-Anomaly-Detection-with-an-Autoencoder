#importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#loading the mnist dataset 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#normalizing the data 
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#reshaping the data 
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#defining a normal class and an anomaly class 
normal_class = 1
anomaly_class = 8

#filtering the normal data 
x_train_normal = x_train[y_train == normal_class]
x_test_normal = x_test[y_test == normal_class]

#filtering the anomaly data 
x_test_anomaly = x_test[y_test == anomaly_class]

#visualizing the normal data 
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.imshow(x_train_normal[i].reshape(28, 28), cmap='gray')
#     plt.axis('off')
# plt.show()

#visualizing the anomaly data 
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.imshow(x_test_anomaly[i].reshape(28, 28), cmap='gray')
#     plt.axis('off')
# plt.show()

encoding_dim = 32

#input 
input_img = tf.keras.layers.Input(shape=(784,))

#encoded representation of the input 
encoded = tf.keras.layers.Dense(128, activation='relu')(input_img)

#2nd encoding 
encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)

#3rd encoding 
encoded_bottleneck = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoded)

#decoded representation of the input 
decoded = tf.keras.layers.Dense(64, activation='relu')(encoded_bottleneck)

#2nd decoding 
decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)

#3rd decoding 
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(decoded)

#autoencoder
autoencoder = tf.keras.models.Model(input_img, decoded)

#encoder
encoder = tf.keras.models.Model(input_img, encoded_bottleneck)

#compiling the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

#summary of the autoencoder 
autoencoder.summary()

#training the autoencoder 
history = autoencoder.fit(x_train_normal, x_train_normal,
                epochs=20,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_normal, x_test_normal))

#plotting the loss 
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.show()

#predicting the reconstructed data 
x_test_normal_recon = autoencoder.predict(x_test_normal)

#predicting the anomaly data 
x_test_anomaly_recon = autoencoder.predict(x_test_anomaly)

#calculating the reconstruction error for normal data 
mse_normal = tf.keras.losses.MeanSquaredError()(x_test_normal, x_test_normal_recon).numpy()

#calculating the reconstruction error for anomaly data 
mse_anomaly = tf.keras.losses.MeanSquaredError()(x_test_anomaly, x_test_anomaly_recon).numpy()

#plotting the reconstruction error 
# plt.hist(mse_normal, bins=50, alpha=0.5, label='normal')
# plt.hist(mse_anomaly, bins=50, alpha=0.5, label='anomaly')
# plt.xlabel('reconstruction error')
# plt.ylabel('number of samples')
# plt.legend(loc='upper right')
# plt.show()

#plot the original and reconstructed images for normal data 
# plt.figure(figsize=(10, 10))

#plotting the original images in the top row 
# for i in range(10):
#     plt.subplot(2, 10, i+1)
#     plt.imshow(x_test_normal[i].reshape(28, 28), cmap='gray')
#     plt.title('original')
#     plt.axis('off')

#plotting the reconstructed images in the bottom row 
# for i in range(10):
#     plt.subplot(2, 10, i+11)
#     plt.imshow(x_test_normal_recon[i].reshape(28, 28), cmap='gray')
#     plt.title('rs')
#     plt.axis('off')

# plt.show()

#threshold
threshold = np.max(mse_normal)
print('MSE threshold:', threshold)
print('Number of anomaly samples:', len(mse_anomaly[mse_anomaly > threshold]))

#plotting the distribution of the reconstruction error for normal and anomaly data 
# plt.hist(mse_normal, bins=50, alpha=0.5, label='normal')
# plt.hist(mse_anomaly, bins=50, alpha=0.5, label='anomaly')
# plt.xlabel('reconstruction error')
# plt.ylabel('number of samples')
# plt.legend(loc='upper right')
# plt.axvline(x=threshold, color='r', linestyle='--')
# plt.show()

#plotting original vs reconstructed images for anomaly data 
plt.figure(figsize=(10, 10))

#plotting the original images in the top row 
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test_anomaly[i].reshape(28, 28), cmap='gray')
    plt.title('original')
    plt.axis('off')
    
#plotting the reconstructed images in the bottom row 
for i in range(10):
    plt.subplot(2, 10, i+11)
    plt.imshow(x_test_anomaly_recon[i].reshape(28, 28), cmap='gray')
    plt.title('rs')
    plt.axis('off')

plt.show()
