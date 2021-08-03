from functions.All_functions import *
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from functions.classVis import IntegratedGradients
from resnet3d import Resnet3DBuilder
import logging

# Set logger
logging.basicConfig(filename = 'Model_perfomance.log', level = logging.INFO, format = '%(asctime)s %(filename)s %(levelname)s %(message)s')

# File directory
cwd = os.getcwd()
file_path = './filePath'
IG_save_path = './savePath'

# Setting variables for integrated gradient
m_steps=50
baseline = tf.zeros(shape=(64, 64, 64, 1), dtype = 'float32')
#baseline = tf.convert_to_tensor(test_images[5], dtype= 'float32')
alphas = tf.linspace(start=0.0, stop=1, num=m_steps+1) # Generate m_steps intervals for integral_approximation() below.


# Set test images
test_images = myreadfile_pad(file_path, 64)[1]
test_images = test_images[...,None]
test_labels = np.ones(len(test_images))
print(f'shape of images: {test_images.shape}')
print(f'labels: {test_labels}')

# set basic model and the path of its weights

weights_dict = {
    'weights_1' : './weights_1',
    'weights_2' : './weights_2.hdf5',
    'weights_3' : './weights_3.hdf5'
                }

model = Resnet3DBuilder.build_resnet_50((64, 64, 64, 1), 2)

os.chdir('/Users/MRILab/Desktop/roc_fig')
plt.style.use('seaborn')
logging.info('Performance of ResNet_50')
i =1
for key in weights_dict:
    model.load_weights(weights_dict[key])
    pred_prob = model.predict(test_images)
    pred_trans = np.argmax(pred_prob, axis = 1)
    fpr, tpr, thresh = roc_curve(test_labels, pred_prob[:, 1])
    auc_score = roc_auc_score(test_labels, pred_prob[:, 1])
    accu = accuracy_score(test_labels, pred_trans)
    recall = recall_score(test_labels, pred_trans)
    precision = precision_score(test_labels, pred_trans)
    f1 = f1_score(test_labels, pred_trans)
    logging.info(f'Weights: {weights_dict[key]}, predict_result: {pred_trans},fpr: {fpr}, tpr: {tpr}, auc: {auc_score}, accuracy: {accu}, prcision: {precision}, recall: {recall}, f1_score: {f1}')
    plt.plot(fpr, tpr, linestyle = '-', label = f"ResNet50_{i} auc({auc_score:.2f})")
    i += 1
    plt.plot([0 , 1], [0, 1], linestyle = '--')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc = 'lower right')
    plt.savefig("Res50_roc.png", dpi = 500)
    plt.close()


# IG plot
for key in weights_dict:
    model.load_weights(weights_dict[key])
    output_dir = IG_save_path + "/Res" + key
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir(output_dir)
    background = test_images[0]
    target = 1

    ig_result = IntegratedGradients(model = model, dataset = test_images, target_index = 1)
    ig_result = ig_result.calculate_IG(img_shape = test_images[0].shape, m_steps = 50)

    mean_ig = np.mean(ig_result, axis = 0)
    mean_ig = data_preprocessing(mean_ig)
    #logging.info(f"dimension:{mean_ig.ndim}", f"shape: {mean_ig.shape}", f"max ig:{mean_ig.max()}",f"min ig:{mean_ig.min()}")

    for j in range(64):
        t = background[:,:,j,0]
        t.astype('float32')
        g = mean_ig[:,:,j, 0]
        overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
        plt.subplot(8, 8, j+1)
        plt.imshow(overlay, cmap='jet')
        plt.axis('off')
    plt.savefig(f'ResNet_targetc+_mean_axial.png')
    plt.close()

    for j in range(64):
        t = background[:,j,:,0]
        t.astype('float32')
        g = mean_ig[:,j,:, 0]
        overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
        plt.subplot(8, 8, j+1)
        plt.imshow(overlay, cmap='jet')
        plt.axis('off')
    plt.savefig(f'ResNet_targetc+_mean_coronal.png')
    plt.close()

    for j in range(64):
        t = background[j,:,:,0]
        t.astype('float32')
        g = mean_ig[j,:,:, 0]
        overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
        plt.subplot(8, 8, j+1)
        plt.imshow(overlay, cmap='jet')
        plt.axis('off')
    plt.savefig(f'ResNet_targetc+_mean_sagittal.png')
    plt.close()
  