from Model_Code.Final_Models import build_generator, build_critic

generator = build_generator(input_shape=(256, 256, 3))
discriminator = build_critic(input_shape=(256, 256, 3))
validation_input = "C:\\Users\\bdoms\\DeburGAN_Capstone\\Datasets\\Validation\\Inputs"

validation_output = "C:\\Users\\bdoms\\DeburGAN_Capstone\\Datasets\\Validation\\Outputs"

dataGen = DataLoader2(validation_input, validation_target,
                      crop=(256, 256), num_folders=1, batch_size=5)
validGAN = GAN(generator, discriminator,
               DataGenerator=dataGen, load_model=True)
validGAN.create_dataset()
validGAN.show_latest_checkpoint(5)
