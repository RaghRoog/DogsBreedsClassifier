import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D,Lambda, Dropout, InputLayer, Input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import warnings


warnings.filterwarnings('ignore')

train_dir = 'train'
test_dir = 'test'

train_size = len(os.listdir(train_dir))
test_size = len(os.listdir(test_dir))

print('Train size:', train_size, ' Test size:', test_size)

df = pd.read_csv('labels.csv')
print(df.head())

dog_breeds = sorted(df['breed'].unique())
n_classes = len(dog_breeds)
print(n_classes)
print(dog_breeds)
# Zapisanie dog_breeds do pliku tekstowego
with open('dog_breeds.txt', 'w') as f:
    for item in dog_breeds:
        f.write("%s\n" % item)

#Konwertowanie klas na liczby
class_to_num = dict(zip(dog_breeds,range(n_classes)))

#Funkcja do ładowania i konwertowania zdjęć na tablice
def images_to_array(data_dir, df, image_size):
    image_names = df['id']
    image_labels = df['breed']
    data_size = len(image_names)

    X = np.zeros([data_size, image_size[0], image_size[1], image_size[2]], dtype=np.uint8)
    y = np.zeros([data_size, 1], dtype=np.uint8)

    for i in range(data_size):
        img_name = image_names[i]
        img_dir = os.path.join(data_dir, img_name + '.jpg')
        img_pixels = load_img(img_dir, target_size=image_size)
        X[i] = img_pixels
        y[i] = class_to_num[image_labels[i]]

    y = to_categorical(y)

    ind = np.random.permutation(data_size)
    X = X[ind]
    y = y[ind]
    print('Ouptut Data Size: ', X.shape)
    print('Ouptut Label Size: ', y.shape)
    return X, y

#Ustalanie rozmiaru zdjęć zgodnie z pretrenowanymi modelami
img_size = (299,299,3)
X, y = images_to_array(train_dir,df,img_size)

#Funkcja do wyciąganie cech z obrazów
def get_features(model_name, data_preprocessor, input_size, data):
    #Prepare pipeline.
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs = avg)
    #Extract feature.
    feature_maps = feature_extractor.predict(data, batch_size=32, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)
    return feature_maps

#Wyciąganie cech za pomocą InceptionV3
inception_preprocessor = preprocess_input
inception_features = get_features(InceptionV3,
                                  inception_preprocessor,
                                  img_size, X)

#Wyciąganie cech za pomocą Xception
xception_preprocessor = preprocess_input
xception_features = get_features(Xception,
                                 xception_preprocessor,
                                 img_size, X)

#Wyciąganie cech za pomocą InceptionResnetV2
inc_resnet_preprocessor = preprocess_input
inc_resnet_features = get_features(InceptionResNetV2,
                                   inc_resnet_preprocessor,
                                   img_size, X)

#Łączenie cech
final_features = np.concatenate([inception_features,
                                 xception_features,
                                 inc_resnet_features,], axis=-1)
print('Final feature maps shape', final_features.shape)

del X

#Callbacki
EarlyStop_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
my_callback=[EarlyStop_callback]

#Budowanie modelu
model = Sequential()
model.add(InputLayer(final_features.shape[1:]))
model.add(Dropout(0.7))
model.add(Dense(120,activation='softmax'))

#Kompilowanie modelu
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

#Trenowanie modelu
history = model.fit(final_features,
                  y,
                  batch_size=32,
                  epochs=50,
                  validation_split=0.1,
                  callbacks=my_callback)

model.save('model.h5')

# Rysowanie wykresów accuracy i val_accuracy oraz loss i val_loss
def plot_training_history(history):
    # Rysowanie wykresu accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Rysowanie wykresu loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Wywołanie funkcji do rysowania wykresów
plot_training_history(history)