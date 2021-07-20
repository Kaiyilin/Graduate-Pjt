class integrated_gradient():
    
    def __init__(self):
        pass


    def interpolate_images(baseline, image, alphas):
        import tensorflow as tf
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
        import tensorflow as tf
        with tf.GradientTape() as tape:
            tape.watch(images)
            logits = model(images)
            probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
        return tape.gradient(probs, images)

    def integral_approximation(gradients):
        import tensorflow as tf
        # riemann_trapezoidal
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients


