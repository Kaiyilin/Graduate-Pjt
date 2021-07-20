import pymongo
from functions.All_functions import *
from functions.cm_tensorboard import *
from functions.classVis import IntegratedGradients
from resnet3d_variation import ResnetSE_3DBuilder

_lr = 1e-3
_mongo_auth = ""
_opt_sgd = tf.keras.optimizers.SGD(lr= _lr, momentum=0.9, nesterov=True)
_loss_func = tf.keras.losses.CategoricalCrossentropy()
_buffer_size = 120
_batch_size = 5
_num_classes = 3
db_name = ""
collection_name = ""

client = pymongo.MongoClient(_mongo_auth)
mydb = client[db_name]
mycollection = mydb[collection_name]

docs = mycollection.find({},{"Images.mfalff":1})  

imgs = []
for doc in docs: 
    img = np.array(doc["Images"]["mfalff"]) 
    img = img[None, ...] 
    imgs.append(img) 
imgs = np.concatenate(imgs, axis = 0)
img_labels = np.zeros(len(imgs))

strategy = tf.distribute.MirroredStrategy()
print('\nNumber of GPU devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    os.chdir(logdir)
    model = ResnetSE_3DBuilder.build_resnet_50(input_shape = imgs[0].shape, num_outputs = _num_classes)
    print("\nBuilding a Res_Net50 model")
    model.compile(optimizer=  _opt_sgd, loss= _loss_func)
    print(model.summary())
                
    callbacks = [ 
                tf.keras.callbacks.TensorBoard(log_dir=logdir, 
                                                    histogram_freq=1, 
                                                    write_graph=True, 
                                                    write_images=False,
                                                    update_freq='epoch', 
                                                    profile_batch=2, 
                                                    embeddings_freq=0,
                                                    embeddings_metadata=None),
                tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                   verbose=0,
                                                   save_weights_only=True,
                                                   save_freq='epoch')
                ]

    history = model.fit()

ig_result = IntegratedGradients(model = model, dataset = imgs, target_index = 1)
ig_result = ig_result.calculate_IG(img_shape = imgs[0].shape, m_steps = 50)

