from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_v3_pre
from keras.applications.xception import Xception, preprocess_input as xception_pre
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as inception_resnet_v2_pre
from keras.models import Model
from keras.layers import Lambda, GlobalAveragePooling2D, Input
import numpy as np

class Predicter:


    def predict(self, image_path):
        # Wczytanie modelu
        model = load_model('model.h5')

        # Wczytanie dog_breeds z pliku tekstowego
        with open('dog_breeds.txt', 'r') as f:
            dog_breeds = f.read().splitlines()
        # Wczytanie i przetworzenie obrazu
        image = load_img(image_path, target_size=(299, 299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # Ekstrakcja cech za pomocą tych samych modeli, które były używane podczas treningu
        def get_features(model_name, preprocessor, input_size, data):
            input_layer = Input(input_size)
            preprocessor = Lambda(preprocessor)(input_layer)
            base_model = model_name(weights='imagenet', include_top=False, input_shape=input_size)(preprocessor)
            avg = GlobalAveragePooling2D()(base_model)
            feature_extractor = Model(inputs=input_layer, outputs=avg)
            feature_maps = feature_extractor.predict(data, batch_size=32, verbose=1)
            return feature_maps

        img_size = (299,299,3)
        inception_features = get_features(InceptionV3, inception_v3_pre, img_size, image)
        xception_features = get_features(Xception, xception_pre, img_size, image)
        inc_resnet_features = get_features(InceptionResNetV2, inception_resnet_v2_pre, img_size, image)

        # Łączenie cech
        final_features = np.concatenate([inception_features, xception_features, inc_resnet_features,], axis=-1)

        # Przewidywanie rasy psa
        preds = model.predict(final_features)
        breed = np.argmax(preds, axis=1)

        # Ustalenie progu pewności
        threshold = 0.5

        # Wyświetlenie przewidzianej rasy psa i prawdopodobieństwa
        if np.max(preds) > threshold:
            return f'Przewidziana rasa psa to: {dog_breeds[breed[0]]}, \nPrawdopodobieństwo: {round(np.max(preds)*100, 2)}%'
        else:
            return 'Nie jestem pewien rasy tego psa.'

