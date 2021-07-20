from functions.All_functions import *
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from datetime import datetime
import logging

logging.basicConfig(filename = 'Model_perfomance.log', level = logging.INFO, format = '%(asctime)s %(filename)s %(levelname)s %(message)s')
cwd = os.getcwd()

# File directory
file_path = '/Users/MRILab/Desktop/BrCA_Map_2/c+/mfalff/'
file_path_2 = '/Users/MRILab/Desktop/BrCA_Map_2/hc/mfalff/'
bg_path = '/Users/MRILab/Desktop/mask_folder/'
IG_save_path = '/Users/MRILab/Desktop/IG_performance'

# Setting test images
images = myreadfile_pad(file_path, 64)[1][:5]
images_2 = myreadfile_pad(file_path_2, 64)[1][:5]
test_images = np.concatenate([images, images_2])
test_images = test_images[...,None]
test_labels = np.concatenate([np.ones(5), np.zeros(5)])
del images, images_2
print(f'shape of images: {test_images.shape}')
print(f'labels: {test_labels}')

# Setting variables for integrated gradient
m_steps=50
baseline = tf.zeros(shape=(64, 64, 64, 1), dtype = 'float32')
#baseline = tf.convert_to_tensor(test_images[5], dtype= 'float32')
alphas = tf.linspace(start=0.0, stop=1, num=m_steps+1) # Generate m_steps intervals for integral_approximation() below.

weights_dict = {'weights_1' : '/Users/MRILab/Desktop/trckpt/20200915-220719/weights-179.hdf5',
                'weights_2' : '/Users/MRILab/Desktop/trckpt/20201006-130022/weights-159.hdf5',
                'weights_3' : '/Users/MRILab/Desktop/trckpt/20201006-160018/weights-259.hdf5',
                'weights_4' : '/Users/MRILab/Desktop/trckpt/20201022-131216/weights-250.hdf5',
                'weights_5' : '/Users/MRILab/Desktop/trckpt/20201022-131243/weights-226.hdf5'}

weights_dict_2 = {'weights_1' : '/Users/MRILab/Desktop/trckpt/20201018-212520/weights-171.hdf5',
                  'weights_2' : '/Users/MRILab/Desktop/trckpt/20201018-212614/weights-159.hdf5',
                  'weights_3' : '/Users/MRILab/Desktop/trckpt/20201018-225742/weights-159.hdf5',
                  'weights_4' : '/Users/MRILab/Desktop/trckpt/20201018-225739/weights-200.hdf5',
                  'weights_5' : '/Users/MRILab/Desktop/trckpt/20201025-180109/weights-223.hdf5'}

# IG functions
def interpolate_images(baseline, image, alphas):

    if image.dtype != 'float32':
        image = tf.convert_to_tensor(image)
        image = tf.dtypes.cast(image, dtype=tf.float32)
    else:
        image = tf.convert_to_tensor(image)

    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta
    return images

def compute_gradients(images, model, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)

def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients

from resnet3d import Resnet3DBuilder

model = Resnet3DBuilder.build_resnet_50((64, 64, 64, 1), 2)

os.chdir('/Users/MRILab/Desktop/roc_fig')
logging.info('Performance of ResNet_50')
plt.style.use('seaborn')

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
    target = 1

    for i in range(5):
        target_img = test_images[i]
        interpolated_images = interpolate_images(baseline = baseline, image = target_img , alphas = alphas)
        path_gradients = compute_gradients(images=interpolated_images, model = model, target_class_idx = target)
        print(path_gradients.shape)
        ig = integral_approximation(gradients=path_gradients)
        ba_list.append(ig)
        print(ig.dtype)
        logging.info(f"Greatest value of ig:{tf.reduce_max(ig)}")

        for j in range(64):
            t = target_img[:,:,j,0]
            t.astype('float32')
            g = ig[:,:,j, 0]
            overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
            plt.subplot(8, 8, j+1)
            plt.imshow(overlay, cmap='jet')
            plt.axis('off')
        plt.savefig(f'ResNet_targetc+_{i+1}_axial.png')
        plt.close()

        for j in range(64):
            t = target_img[:,j,:,0]
            t.astype('float32')
            g = ig[:,j,:, 0]
            overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
            plt.subplot(8, 8, j+1)
            plt.imshow(overlay, cmap='jet')
            plt.axis('off')
        plt.savefig(f'ResNet_targetc+_{i+1}_coronal.png')
        plt.close()

        for j in range(64):
            t = target_img[j,:,:,0]
            t.astype('float32')
            g = ig[j,:,:, 0]
            overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
            plt.subplot(8, 8, j+1)
            plt.imshow(overlay, cmap='jet')
            plt.axis('off')
        plt.savefig(f'ResNet_targetc+_{i+1}_sagittal.png')
        plt.close()

    mean_ig = np.mean(ba_list, axis = 0)
    mean_ig = data_preprocessing(mean_ig)
    #logging.info(f"dimension:{mean_ig.ndim}", f"shape: {mean_ig.shape}", f"max ig:{mean_ig.max()}",f"min ig:{mean_ig.min()}")

    for j in range(64):
        t = target_img[:,:,j,0]
        t.astype('float32')
        g = mean_ig[:,:,j, 0]
        overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
        plt.subplot(8, 8, j+1)
        plt.imshow(overlay, cmap='jet')
        plt.axis('off')
    plt.savefig(f'ResNet_targetc+_mean_axial.png')
    plt.close()

    for j in range(64):
        t = target_img[:,j,:,0]
        t.astype('float32')
        g = mean_ig[:,j,:, 0]
        overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
        plt.subplot(8, 8, j+1)
        plt.imshow(overlay, cmap='jet')
        plt.axis('off')
    plt.savefig(f'ResNet_targetc+_mean_coronal.png')
    plt.close()

    for j in range(64):
        t = target_img[j,:,:,0]
        t.astype('float32')
        g = mean_ig[j,:,:, 0]
        overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
        plt.subplot(8, 8, j+1)
        plt.imshow(overlay, cmap='jet')
        plt.axis('off')
    plt.savefig(f'ResNet_targetc+_mean_sagittal.png')
    plt.close()

    # Plotting hc subjects 
    target = 0
    for i in range(5,10):
        target_img = test_images[i]
        interpolated_images = interpolate_images(baseline = baseline, image = target_img , alphas = alphas)
        path_gradients = compute_gradients(images=interpolated_images, model = model, target_class_idx = target)
        print(path_gradients.shape)
        ig = integral_approximation(gradients=path_gradients)
        print(ig.shape)
        for j in range(64):
            t = target_img[:,:,j,0]
            t.astype('float32')
            g = ig[:,:,j,0].numpy()
            overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
            plt.subplot(8, 8, j+1)
            plt.imshow(overlay, cmap='jet')
            plt.axis('off')
        plt.savefig(f'ResNet_targethc_{i+1}.png')
        plt.close()
        
        for j in range(64):
            t = target_img[:,j,:,0]
            t.astype('float32')
            g = ig[:,j,:, 0].numpy()
            overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
            plt.subplot(8, 8, j+1)
            plt.imshow(overlay, cmap='jet')
            plt.axis('off')
        plt.savefig(f'ResNet_targethc_{i+1}_coronal.png')
        plt.close()

        for j in range(64):
            t = target_img[j,:,:,0]
            t.astype('float32')
            g = ig[j,:,:, 0].numpy()
            overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
            plt.subplot(8, 8, j+1)
            plt.imshow(overlay, cmap='jet')
            plt.axis('off')
        plt.savefig(f'ResNet_targethc_{i+1}_sagittal.png')
        plt.close()
    
from densenet3d.DenseNet import DenseNet3Dbuilder
model_2 =DenseNet3Dbuilder.densenet_121((64,64,64,1), 2, 8)

os.chdir('/Users/MRILab/Desktop/roc_fig')
plt.style.use('seaborn')
i =1
logging.info('Performance of DesNet121_growth_8')

for key in weights_dict_2:
    model_2.load_weights(weights_dict_2[key])
    pred_prob = model_2.predict(test_images)
    pred_trans = np.argmax(pred_prob, axis = 1)
    fpr, tpr, thresh = roc_curve(test_labels, pred_prob[:, 1])
    auc_score = roc_auc_score(test_labels, pred_prob[:, 1])
    accu = accuracy_score(test_labels, pred_trans)
    recall = recall_score(test_labels, pred_trans)
    precision = precision_score(test_labels, pred_trans)
    f1 = f1_score(test_labels, pred_trans)
    logging.info(f'Weights: {weights_dict_2[key]}, predict_result: {pred_trans},fpr: {fpr}, tpr: {tpr}, auc: {auc_score}, accuracy: {accu}, prcision: {precision}, recall: {recall}, f1_score: {f1}')
    plt.plot(fpr, tpr, linestyle = '-', label = f"DenseNet121_{i} auc({auc_score:.2f})")
    i += 1

plt.plot([0 , 1], [0, 1], linestyle = '--')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc = 'lower right')
plt.savefig("Dense121_roc.png", dpi = 500)
plt.close()

for key in weights_dict_2:
    model_2.load_weights(weights_dict_2[key])
    output_dir = IG_save_path + "/Dense" + key
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.chdir(output_dir)
    target = 1
    ba_list = list()
    for i in range(5):
        target_img = test_images[i]
        interpolated_images = interpolate_images(baseline = baseline, image = target_img , alphas = alphas)
        path_gradients = compute_gradients(images=interpolated_images, model = model_2, target_class_idx = target)
        print(path_gradients.shape)
        ig = integral_approximation(gradients=path_gradients)
        ba_list.append(ig)
        print(ig.dtype)

        for j in range(64):
            t = target_img[:,:,j,0]
            t.astype('float32')
            g = ig[:,:,j, 0].numpy()
            overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
            plt.subplot(8, 8, j+1)
            plt.imshow(overlay, cmap='jet')
            plt.axis('off')
        plt.savefig(f'DenseNet_targetc+_{i+1}_axial.png')
        plt.close()

        for j in range(64):
            t = target_img[:,j,:,0]
            t.astype('float32')
            g = ig[:,j,:, 0].numpy()
            overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
            plt.subplot(8, 8, j+1)
            plt.imshow(overlay, cmap='jet')
            plt.axis('off')
        plt.savefig(f'DenseNet_targetc+_{i+1}_coronal.png')
        plt.close()

        for j in range(64):
            t = target_img[j,:,:,0]
            t.astype('float32')
            g = ig[j,:,:, 0].numpy()
            overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
            plt.subplot(8, 8, j+1)
            plt.imshow(overlay, cmap='jet')
            plt.axis('off')
        plt.savefig(f'DenseNet_targetc+_{i+1}_sagittal.png')
        plt.close()

    mean_ig = np.mean(ba_list, axis = 0)

    for j in range(64):
        t = target_img[:,:,j,0]
        t.astype('float32')
        g = mean_ig[:,:,j, 0]
        overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
        plt.subplot(8, 8, j+1)
        plt.imshow(overlay, cmap='jet')
        plt.axis('off')
    plt.savefig(f'DenseNet_targetc+_mean_axial.png')
    plt.close()

    for j in range(64):
        t = target_img[:,j,:,0]
        t.astype('float32')
        g = mean_ig[:,j,:, 0]
        overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
        plt.subplot(8, 8, j+1)
        plt.imshow(overlay, cmap='jet')
        plt.axis('off')
    plt.savefig(f'DenseNet_targetc+_mean_coronal.png')
    plt.close()

    for j in range(64):
        t = target_img[j,:,:,0]
        t.astype('float32')
        g = mean_ig[j,:,:, 0]
        overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
        plt.subplot(8, 8, j+1)
        plt.imshow(overlay, cmap='jet')
        plt.axis('off')
    plt.savefig(f'DenseNet_targetc+_mean_sagittal.png')
    plt.close()

    # Plotting hc subjects 
    target = 0
    for i in range(5,10):
        target_img = test_images[i]
        interpolated_images = interpolate_images(baseline = baseline, image = target_img , alphas = alphas)
        path_gradients = compute_gradients(images=interpolated_images, model = model_2, target_class_idx = target)
        print(path_gradients.shape)
        ig = integral_approximation(gradients=path_gradients)
        print(ig.shape)
        for j in range(64):
            t = target_img[:,:,j,0]
            t.astype('float32')
            g = ig[:,:,j,0].numpy()
            overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
            plt.subplot(8, 8, j+1)
            plt.imshow(overlay, cmap='jet')
            plt.axis('off')
        plt.savefig(f'DenseNet_targethc_{i+1}.png')
        plt.close()
        
        for j in range(64):
            t = target_img[:,j,:,0]
            t.astype('float32')
            g = ig[:,j,:, 0].numpy()
            overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
            plt.subplot(8, 8, j+1)
            plt.imshow(overlay, cmap='jet')
            plt.axis('off')
        plt.savefig(f'DenseNet_targethc_{i+1}_coronal.png')
        plt.close()

        for j in range(64):
            t = target_img[j,:,:,0]
            t.astype('float32')
            g = ig[j,:,:, 0].numpy()
            overlay = cv2.addWeighted(t, 0.3, g, 0.7, 0, dtype = cv2.CV_32F)
            plt.subplot(8, 8, j+1)
            plt.imshow(overlay, cmap='jet')
            plt.axis('off')
        plt.savefig(f'DenseNet_targethc_{i+1}_sagittal.png')
        plt.close()
