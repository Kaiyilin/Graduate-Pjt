import sys
import numpy as np

# Data Preprocessing
def data_preprocessing(image):
    image = (image - image.min())/(image.max() - image.min()) 
    return image

def standardised(image): 
    img = (image - image.mean())/image.std() 
    return img 

def pad_to_sqr(array, pad_size, channel_last=True):
    
    elements = array.shape    
    for element in elements:
        if element > pad_size:
            sys.exit('\nThe expanded dimension shall be greater than your current dimension')
    pad_list = list() 
    if channel_last == True:
        for i in range(array.ndim - 1):
            x = pad_size - array.shape[i]
            if x%2 ==1:
                y_1 = (x/2 +0.5)
                y_2 = (x/2 -0.5)
                z = (int(y_1),int(y_2))
                pad_list.append(z)

            else:
                y = int(x/2)
                z=(y,y)
                pad_list.append(z)
    pad_list.append((0,0))
    pad_array = np.pad(array, pad_list, 'constant')
    pad_list = list() 
    return pad_array
