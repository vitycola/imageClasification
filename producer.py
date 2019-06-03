
# coding: utf-8

# # Lectura y envío de imágenes

# Importar librerias

# In[17]:


from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
from kafka import KafkaProducer
from kafka.errors import KafkaError
from resizeimage import resizeimage
import time
import json


# Leer cada imagen

# In[2]:


def readImage(rutaImagen):
#Recibe una ruta de una imagen y la abre en su formato correspondiente
    image=Image.open(rutaImagen).convert("L")
    return image


# Redimensionar imagen a 48 x 48

# In[3]:


#Recibe una imagen en escala de grises y la redimensiona a 48 x 48 pixeles
def redimensionImage(greyImage):
    redimImage=resizeimage.resize_cover(greyImage, [48,48])
    return redimImage


# Enviar matriz de pixeles a Kafka

# Main

# In[124]:


pathImages = "/Users/victor/PycharmProjects/imagesPrediction/resources/photo/"
# Aqui se debe obtener la lista de imagenes de la ruta que se especifique
listaImagenes = listdir(pathImages)
#Declaramos el productor de Kafka
producer=KafkaProducer(bootstrap_servers=['localhost:9092'])
# Aqui se debe recorrer la la lista de imagenes obtenida en el punto anterior con un bucle
for imagen in listaImagenes:
    # Dentro del bucle, para cada imagen se debe leer la imagen
    imagenLeida = readImage(pathImages + imagen)
    # Una vez pasada a escala de grises la imagen se debe redimensionar a 48 x 48 pixeles
    imageFinal=redimensionImage(imagenLeida)
    # Opcional: mostrar la imagen
    #imageFinal.show()
    # Una vez este la imagen lista se debe enviar a kafka
    #producer.send('topic1',np.array(imageFinal).flatten().tobytes())
    producer.send('topic1',bytes(imageFinal.getdata()))
    # Tiempo de espera de X segundos entre foto y foto
    time.sleep(5) # espera en segundos


# In[ ]:


emotions = {0 : "Enfadado",
            1 : "Disgustado",
            2 : "Asustado",
            3 : "Feliz",
            4 : "Triste",
            5 : "Sorprendido",
            6 : "Neutral"}


# In[50]:


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
x = np.array(imageFinal).flatten().tobytes()


# In[74]:


a = np.array(imageFinal).flatten()
b = a.tobytes()
c = np.frombuffer(a,dtype = "uint8")


# In[75]:


c


# In[114]:


x = bytes(imageFinal.getdata())


# In[115]:


type(x)


# In[123]:


list(x)

