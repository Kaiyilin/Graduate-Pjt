# Set a callback for the validate data 
class evolRecord(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    predict_img = self.model.generator.predict(ds.take(1).batch(1))
    plt.imshow(predict_img[0, :, :, 64, 0], cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(bst.checkpoint_dir, "PrdctImg_Epoch{}.png".format(epoch + 1)), dpi=72)