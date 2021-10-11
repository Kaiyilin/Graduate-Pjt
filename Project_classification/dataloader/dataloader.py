import pymongo
import numpy as np
import tensorflow as tf
 

_mongo_auth = ""
db_name = ""
collection_name = ""

client = pymongo.MongoClient(_mongo_auth)
mydb = client[db_name]
mycollection = mydb[collection_name]
docs = mycollection.find({},{"Images.mfalff":1, "Labels":1})  

imgs = []
labels = []
for doc in docs: 
    img = np.array(doc["Images"]["mfalff"]) 
    labels.append(doc["Labels"])
    img = img[None, ...] 
    imgs.append(img) 
imgs = np.concatenate(imgs, axis = 0)

# create tf data dataset
ds = tf.data.Dataset.from_tensor_slices(imgs, labels)
