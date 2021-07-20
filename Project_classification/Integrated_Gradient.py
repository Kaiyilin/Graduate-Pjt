# Integrated Gradient
from functions.All_functions import *

baseline = tf.zeros(shape=(64, 64, 64, 1), dtype = 'float32')
image = tf.ones(shape=(64, 64, 64, 1), dtype = 'float32')

m_steps=50
alphas = tf.linspace(start=0.0, stop=0.5, num=m_steps+1) # Generate m_steps intervals for integral_approximation() below.

def interpolate_images(baseline, image, alphas):

    if image.dtype != 'float32':
        image = tf.convert_to_tensor(image)
        image = tf.dtypes.cast(image, dtype=tf.float32)
    else:
        pass

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

interpolated_images = interpolate_images(baseline = baseline, image = image , alphas = alphas)

path_gradients = compute_gradients(images=interpolated_images, target_class_idx = 1)
print(path_gradients.shape)

def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients

ig = integral_approximation(gradients=path_gradients)
print(ig.shape)

#############
@tf.function
def integrated_gradients(baseline,
                         image,
                         model,
                         target_class_idx,
                         m_steps=50,
                         batch_size=5):
  # 1. Generate alphas.
  alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

  # Initialize TensorArray outside loop to collect gradients.    
  gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)
    
  # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
  for alpha in tf.range(0, len(alphas), batch_size):
    from_ = alpha
    to = tf.minimum(from_ + batch_size, len(alphas))
    alpha_batch = alphas[from_:to]

    # 2. Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                       image=image,
                                                       alphas=alpha_batch)

    # 3. Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                       model = model, 
                                       target_class_idx=target_class_idx)
    
    # Write batch indices and gradients to extend TensorArray.
    gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    
  
  # Stack path gradients together row-wise into single tensor.
  total_gradients = gradient_batches.stack()

  # 4. Integral approximation through averaging gradients.
  avg_gradients = integral_approximation(gradients=total_gradients)

  # 5. Scale integrated gradients with respect to input.
  integrated_gradients = (image - baseline) * avg_gradients

  return integrated_gradients