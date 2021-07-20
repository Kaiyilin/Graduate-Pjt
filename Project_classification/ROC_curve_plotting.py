# ROC curve plotting
from functions.All_functions import *


def split_and_channel(c,array):
    array_val = array[:c,:,:,:]
    array_tr = array[c:,:,:,:]

    array_tr = array_tr[...,None]
    array_val = array_val[...,None]
    return array_tr, array_val

weights_path = '/home/user/venv/kaiyi_venv/model_preserved/Res_50_BAHC_weights-179.hdf5'

BA_alff, _, HC_alff, _, _, _ = importdata2(dir['BA'],dir['BB'],dir['HC'],dir['BA2'],dir['BB2'],dir['HC2'],64)

BA_alff_tr, BA_alff_val = split_and_channel(5,BA_alff)
HC_alff_tr, HC_alff_val = split_and_channel(5,HC_alff)
del BA_alff, HC_alff

BA_labels_tr = np.ones(BA_alff_tr.shape[0])
BA_labels_val = np.ones(BA_alff_val.shape[0])

HC_labels_tr = np.zeros(HC_alff_tr.shape[0])
HC_labels_val = np.zeros(HC_alff_val.shape[0])

tr_val_images = np.concatenate([BA_alff_tr, HC_alff_tr])
tr_val_labels = np.concatenate([BA_labels_tr, HC_labels_tr])

test_images = np.concatenate([BA_alff_val, HC_alff_val])
test_labels = np.concatenate([BA_labels_val, HC_labels_val])

from resnet3d import Resnet3DBuilder

model = Resnet3DBuilder.build_resnet_50((64,64,64,1), 2)
model.load_weights(weights_path)

predictions = model.predict(tr_val_images)

from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(tr_val_labels, predictions[:,1], pos_label=1)


# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(tr_val_labels))]
p_fpr, p_tpr, _ = roc_curve(tr_val_labels, random_probs, pos_label=1)

from sklearn.metrics import roc_auc_score

# auc scores
auc_score = roc_auc_score(tr_val_labels, predictions[:,1])
print(auc_score)

plt.style.use('seaborn')

img_labels = f'ResNet_50 (auc score = {auc_score:.2f})'
# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label= img_labels)
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')

# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='lower right')

os.chdir('/home/user/venv/kaiyi_venv/Project_classification')
plt.savefig('ROC_tr_val.png',dpi=500)
plt.close()

#################### for test images ############################

predictions = model.predict(test_images)

from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(test_labels, predictions[:,1], pos_label=1)
print(fpr1, tpr1, thresh1)
# auc scores
auc_score = roc_auc_score(test_labels, predictions[:,1])
print(auc_score)

plt.style.use('seaborn')

img_labels = f'ResNet_50 (auc score = {auc_score:.2f})'
# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label= img_labels)
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')

# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='lower right')

os.chdir('/home/user/venv/kaiyi_venv/Project_classification')
plt.savefig('ROC_tr_test.png',dpi=500)
plt.close()

