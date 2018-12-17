import tensorflow as tf
from tf_backend import *

class synthesizer(object):
    def __init__(self):
        """
        Function to initialize the synthesizer class.
        Will see what parameters need to be initialized here.
        Should initialize:
        1) modes and epochs, load from config. 
        2) Placeholders
        3) Generator and discriminator/critic models.

        """
        self.synth_mode = 0
        self.model_mode = 0
        self.epochs = 0
        self.generator = None
        self.critic = None
        self.placeholders = None

    def read_input_file(self, file_name, synth_mode):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on maode. 
        Mode 0 is for direct synthesis from frame-wise phoneme and note annotations, like in Nitech.
        Mode 1 is for partial synthesis, from frame-wise phoeneme and f0 annotations, like in NUS.
        Mode 2 is for sem-partial synthesis, from loose framewise phoneme and note annotation, like in Ayesha recordings.
        Mode 3 is from indirect synthesis, etracting features from the audio recording and syntthesizing. 
        """

    def synth_file(self, input_features):
        """
        Function to synthesize a singing voice based on the input features. 
        """
    def get_placeholders(self):
        """
        Returns the placeholders for the model. 
        Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
        """
    def load_model(self):
        """
        Load model parameters, for synthesis or re-starting training. 
        """
    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """
    def get_optimizers(self, loss_functions):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """
    def get_summary(self, loss_functions):
        """
        Gets the summaries and sumary writers for the losses.
        """
    def save_model(self):
        """
        Save the model.
        """
    def print_summary(self):
        """
        Print training summary to console, every N epochs.
        Summary will depend on model_mode.
        """
    def train(self, model_directory):
        """
        Function to train the model, and save Tensorboard summary, for N epochs. 
        """

    def train_generator(self):
        """
        Function to train the generator for each epoch
        """

    def train_critic(self):
        """
        Function to train the critic for each epoch.
        Will also take care of multiple training loops for Wasserstien GAN.
        """
    def train_discriminator(self):
        """
        Function to train discriminator for each loop.
        """
    def generator(self):
        """
        The main generator function, takes and returns tensors.
        Will be defined in modules.
        """
    def discriminator(self):
        """
        The main discriminator function.
        Will be defined in modules.
        """
    def critic(self):
        """
        The main critic function. 
        Will be defined in modules.
        """
    def phoneme_model(self):
        """
        The phoneme model. 
        Will be defined in modules.
        """
    def singer_model(self):
        """
        The singer/speaker model.
        """






