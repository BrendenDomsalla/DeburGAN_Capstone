from Model_Code.Final_Models import build_critic, build_generator
import tensorflow as tf
from util.DataLoader import DataLoader2
from Model_Code.GAN import GAN

blurred = "C:\\Users\\bdoms\\DeburGAN_Capstone\\Datasets\\Train\\blurred"
sharp = "C:\\Users\\bdoms\\DeburGAN_Capstone\\Datasets\\Train\\sharp"
generator = build_generator(input_shape=(256, 256, 3))
discriminator = build_critic(input_shape=(256, 256, 3))
dataGen = DataLoader2(blurred, sharp, crop=(
    256, 256), num_folders=1, batch_size=5)
gen_optim = tf.keras.optimizers.Adam(1e-4)
disc_optim = tf.keras.optimizers.Adam(1e-4)

gan = GAN(generator, discriminator, gen_optim, disc_optim, crop=(
    256, 256), shuffleData=False, DataGenerator=dataGen, load_model=True)
gan.create_dataset()
# gan.dataset=gan.DataGenerator.create_blank_image_dataset()
gan.show_latest_checkpoint()
gan.train(240)
