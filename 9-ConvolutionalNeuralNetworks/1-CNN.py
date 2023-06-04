from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation="relu")) 

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 1-2 Repeat | 2. Convolution Katmanı
classifier.add(Convolution2D(32, 3, 3, activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - YSA (Yapay Sinir Ağı)
classifier.add(Dense(128, activation="relu"))
classifier.add(Dense(1, activation="sigmoid"))


# CNN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# CNN and Image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale= 1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale= 1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)


training_set = train_datagen.flow_from_directory("data/cnnData/training_set",
                                                target_size=(64, 64),
                                                batch_size=1,
                                                class_mode="binary")            

test_set = test_datagen.flow_from_directory("data/cnnData/test_set",
                                                target_size=(64, 64),
                                                batch_size=1,
                                                class_mode="binary")  


classifier.fit_generator(training_set, steps_per_epoch=8000, 
                        epochs=1,validation_data=test_set)


# Öğrenilen fotoğraflar ile başka fotoğrafları tahmin ettirme aşaması
import pandas as pd
import numpy as np

test_set.reset()
pred = classifier.predict_generator(test_set, verbose=1)

# pred = list(map(round, pred))
pred[pred > .5] = 1
pred[pred <= .5] = 0

testLabels = []

for i in range(0, 203): # resim sayısını yazıyoruz
    testLabels.extend(np.array(test_set[i][1]))


# Dosya İsmi, Tahminler, test = Gerçek Değerler
fileName = test_set.filenames

result = pd.DataFrame()
result["File Name"] = fileName
result["Predictions"] = pred
result["Test"] = testLabels


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(testLabels, pred)
print("Result".center(50, "-"))
print(cm)
print("-".center(50, "-"))