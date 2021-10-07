
#from functions.Basic_tool import bn_relu_block
import sys, os
import tensorflow as tf

class GAN(tf.keras.Model):
    """
    actually it's pix2pix
    """
    def __init__(self, discriminator, generator):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator      

    def compile(self, d_optimizer, 
                g_optimizer, loss_fn: list):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.generator_loss = loss_fn[0]
        self.discriminator_loss = loss_fn[1]

    def train_step(self, data):
        input_image, target = data
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            real_input =tf.keras.layers.concatenate([input_image, target],axis=-1)
            disc_real_output = self.discriminator(real_input, training=True)
            fake_input =tf.keras.layers.concatenate([input_image, gen_output],axis=-1)
            disc_generated_output = self.discriminator(fake_input, training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, 
                                                                            gen_output, target)

            disc_loss = self.discriminator_loss(disc_real_output, 
                                                disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
                                                
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(generator_gradients,
                                             self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(discriminator_gradients,
                                             self.discriminator.trainable_variables))

        return {
            "gen_total_loss":gen_total_loss, "gen_gan_loss": gen_gan_loss, 
            "gen_l1_loss": gen_l1_loss, 'disc_loss': disc_loss
            }

    def summary_all(self):
        """
        Giving the summary of the cycleGAN model
        """
        print(self.generator.summary())
        print(self.discriminator.summary())

    def save_all(self, save_path: str, gen_name: str, disc_name: str, mode = 0):
        """
        save_path: str, given a save path 
        genG_name: str, given a name for generatorG 
        genF_name: str, given a name for generatorF
        disc_x_name: str, given a name for discriminator_x 
        disc_y_name: str, given a name for discriminator_y
        mode: 0 or 1, default 0 for save all the architectures, weights and bias
              1 for saving weights and bias only.
        """
        if mode == 0:
            self.generator.save(os.path.join(save_path, f"{gen_name}.h5"))
            self.discriminator.save(os.path.join(save_path, f"{disc_name}.h5"))

        elif mode == 1:
            self.generator.save_weights(os.path.join(save_path, f"{gen_name}.h5"))
            self.discriminator.save_weights(os.path.join(save_path, f"{disc_name}.h5"))
        
        else:
            sys.exit('The mode should be chosen in either 0 or 1 if you truly like to save models')
    
        # Not finished yet


class cycleGAN_3D(tf.keras.Model):
    """
    suggest you have 2 image domain X and Y
    generatorG convert X to Y
    generatorF convert Y to X

    discriminator_x classify the real and fake in X domain
    discriminator_y classify the real and fake in Y domain
    """
    def __init__(self, discriminator_x, discriminator_y, generatorG, generatorF):
        super(cycleGAN_3D, self).__init__()
        self.discriminator_x = discriminator_x
        self.discriminator_y = discriminator_y
        self.generatorG = generatorG      
        self.generatorF = generatorF

    def compile(self, d_optimizer_x, d_optimizer_y, g_optimizerG, g_optimizerF, loss_fn: list):
        super(cycleGAN_3D, self).compile()
        self.d_optimizer_x = d_optimizer_x
        self.d_optimizer_y = d_optimizer_y
        self.g_optimizerG = g_optimizerG
        self.g_optimizerF = g_optimizerF
        ### 4 kind of loss 
        self.generator_loss = loss_fn[0]
        self.discriminator_loss = loss_fn[1]
        self.cycle_loss = loss_fn[2]
        self.identity_loss = loss_fn[3]


    def train_step(self, data):
        real_x, real_y = data
        with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

            fake_y = self.generatorG(real_x, training=True)
            cycled_x = self.generatorF(fake_y, training=True)

            fake_x = self.generatorF(real_y, training=True)
            cycled_y = self.generatorG(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generatorF(real_x, training=True)
            same_y = self.generatorG(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.cycle_loss(real_x, cycled_x) + self.cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generatorG_gradients = tape.gradient(total_gen_g_loss, 
                                            self.generatorG.trainable_variables)
        generatorF_gradients = tape.gradient(total_gen_f_loss, 
                                            self.generatorF.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.g_optimizerG.apply_gradients(zip(generatorG_gradients, 
                                                self.generatorG.trainable_variables))

        self.g_optimizerF.apply_gradients(zip(generatorF_gradients, 
                                                self.generatorF.trainable_variables))

        self.d_optimizer_x.apply_gradients(zip(discriminator_x_gradients,
                                                    self.discriminator_x.trainable_variables))

        self.d_optimizer_y.apply_gradients(zip(discriminator_y_gradients,
                                                    self.discriminator_y.trainable_variables))

        return {"total_gen_g_loss": total_gen_g_loss, "total_gen_f_loss": total_gen_f_loss, "disc_x_loss": disc_x_loss, "disc_y_loss": disc_y_loss}

    def summary_all(self):
        """
        Giving the summary of the cycleGAN model
        """
        print(self.generatorG.summary())
        print(self.generatorF.summary())
        print(self.discriminator_x.summary())
        print(self.discriminator_y.summary())

    def save_all(self, save_path: str, genG_name: str, 
            genF_name: str, disc_x_name: str, disc_y_name: str, mode = 0):
        """
        save_path: str, given a save path 
        genG_name: str, given a name for generatorG 
        genF_name: str, given a name for generatorF
        disc_x_name: str, given a name for discriminator_x 
        disc_y_name: str, given a name for discriminator_y
        mode: 0 or 1, default 0 for save all the architectures, weights and bias
              1 for saving weights and bias only.
        """
        try:
            os.mkdir(save_path)
        except:
            pass
        
        if mode == 0:
            self.generatorG.save(os.path.join(save_path, f"{genG_name}.h5"))
            self.generatorF.save(os.path.join(save_path, f"{genF_name}.h5"))
            self.discriminator_x.save(os.path.join(save_path, f"{disc_x_name}.h5"))
            self.discriminator_y.save(os.path.join(save_path, f"{disc_y_name}.h5"))
        elif mode ==1:
            self.generatorG.save_weights(os.path.join(save_path, f"weights_{genG_name}.h5"))
            self.generatorF.save_weights(os.path.join(save_path, f"weights_{genF_name}.h5"))
            self.discriminator_x.save_weights(os.path.join(save_path, f"weights_{disc_x_name}.h5"))
            self.discriminator_y.save_weights(os.path.join(save_path, f"weights_{disc_y_name}.h5"))
        else:
            sys.exit('The mode should be chosen in either 0 or 1 if you truly like to save models')
    
    def save_separately(self, save_path: str, genG_name: str, genF_name: str, disc_x_name: str, disc_y_name: str):
        """
        save_path: str, given a save path 
        genG_name: str, given a name for generatorG 
        genF_name: str, given a name for generatorF
        disc_x_name: str, given a name for discriminator_x 
        disc_y_name: str, given a name for discriminator_y
        mode: 0 or 1, default 0 for save all the architectures, weights and bias
              1 for saving weights and bias only.
        """
        # Not finished yet