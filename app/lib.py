
import tensorflow as tf
import keras
from keras.models import load_model
from keras import backend  as K
from keras import optimizers
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import ast
import re
from matplotlib.figure import Figure
import io


def read_data(file):
  dataset = pd.read_csv(file)
  dataset['Input_Sequence']= dataset['Input_Sequence'].apply(lambda x: ast.literal_eval(x))
  dataset['Target_Annotation']= dataset['Target_Annotation'].apply(lambda x: ast.literal_eval(x))
  return dataset

def load_models(model_path):
  #we need the functions used for the masked accuraccy and the loss
  def masked_accuracy(y_true, y_pred):
    '''accuracy that masks based on targets'''
    y = tf.argmax(y_true, axis=-1) 
    y_ = tf.argmax(y_pred, axis=-1)
    mask = tf.greater(y, 0) 
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

  def build_masked_loss(loss_function, mask_value):        
    def masked_loss_function(y_true,y_pred):
      mask = K.cast(K.any(K.not_equal(y_true, mask_value),  axis=-1), K.floatx()) 
      cce = tf.keras.losses.CategoricalCrossentropy()
      loss = cce(tf.boolean_mask(y_true, mask), tf.boolean_mask(y_pred, mask)) 
      return loss

    return masked_loss_function

  mask_value = [1.] + 17*[0.]
  mask_value = np.array(mask_value) 
  opt = optimizers.Adam(lr=0.01)


  model = load_model(model_path, custom_objects={'build_masked_loss': build_masked_loss, 'masked_loss_function':build_masked_loss(K.categorical_crossentropy, mask_value), 'masked_accuracy':masked_accuracy}, compile=False)
  model.compile(optimizer=opt, loss=build_masked_loss(tf.keras.losses.CategoricalCrossentropy, mask_value), metrics=[masked_accuracy])
  print(model.summary())
  return model

def get_tokenizer_encoder_decoder(all_data_file):
  all_data = read_data(all_data_file)
  #prepare amino acid sequences
  tokenizer_encoder = Tokenizer()
  tokenizer_encoder.fit_on_texts(all_data['Input_Sequence'])
  print('Amino acid encoding',tokenizer_encoder.word_index)

  #prepare targets
  tokenizer_decoder = Tokenizer(char_level=True)
  tokenizer_decoder.fit_on_texts(all_data['Target_Annotation'])
  print('Class encoding',tokenizer_decoder.word_index)

  return tokenizer_encoder, tokenizer_decoder


def predict_sequence(seq, model, tokenizer_encoder,tokenizer_decoder):
  ''' predicts the annotation for a given sequence '''
  s = [char for char in seq]
  s1 = preprocess_sequences([s], tokenizer_encoder, 1500) #1000 is max_length
  prediction = list(model.predict_classes(s1)[0])
  reverse_decoder_index = {value:key for key,value in tokenizer_decoder.word_index.items()}
  for k,v in reverse_decoder_index.items():
    reverse_decoder_index.update({k: v.upper()})
  annotation_predicted = [reverse_decoder_index[value] for value in prediction][0:len(seq)]
  annotation_predicted = [x.upper() for x in annotation_predicted]
  return annotation_predicted

def preprocess_sequences(sequences_dataset,tokenizer_encoder,maxlen_seq):
  '''preprocess the sequences of amino acids'''
  #encoding -> assign number to each amino acid 
  input_data = tokenizer_encoder.texts_to_sequences(sequences_dataset)
  #padding -> add zeros in the end to reach maxlen_seq
  input_data = sequence.pad_sequences(input_data, maxlen=maxlen_seq, padding='post')
  #one-hot encoding
  input_data = to_categorical(input_data)
  return input_data

def classes_to_secondary_structure(prediction):
  classes_to_secondary_structure_dict = {'P-S':'h',
  'L-S':'h',
  'C':'l',
  'N':'l',
  'L4':'l',
  'P-I':'h',
  'C-S':'h',
  'L-I':'h',
  'L1':'l',
  'L5':'l',
  'L2':'l',
  'L3':'l',
  'C-I':'h',
  'L7':'l',
  'L6':'l',
  'CYTO':'h',
  'PERI':'h'}
  result = [classes_to_secondary_structure_dict[value] for value in prediction]
  return result

def visualize_dataframe_prediction(sequence, prediction):
  def column_style(val):   
      if val == 'CYTO':
        return 'background-color: #39A2FE'
      if val == 'PERI':
        return 'background-color: #7334A5'
      if val == 'L-S':
        return 'background-color: #F9F147'
      if val == 'L-I':
        return 'background-color: #FBB619'
      if val == 'P-S':
        return 'background-color: #6ADB74'
      if val == 'P-I':
        return 'background-color: #3E8545'
      if val == 'C-S':
        return 'background-color: #F8252B'
      if val == 'C-I':
        return 'background-color: #FAD1B7'
      else:
        return 'background-color: #F8F8F8'
  df = pd.DataFrame()
  df['Amino Acid'] = [c for c in sequence]
  df['Annotation'] = prediction
  df_t = df.T
  #plot_legend()
  #return df_t.style.applymap(column_style)
  return df_t


def print_helix_location(annotation):
    helix_indices = [i for i, x in enumerate(annotation) if x == "h"]
    helix_indices = sorted(set(helix_indices))
    gaps = [[s, e] for s, e in zip(helix_indices, helix_indices[1:]) if s+1 < e]
    edges = iter(helix_indices[:1] + sum(gaps, []) + helix_indices[-1:])
    helices_list = list(zip(edges, edges))
    helices_names = []
    for i in range(len(helices_list)):
      helices_names.append( "Helix " + str(i+1) + ":")
      print("Helix " + str(i+1) + ":" + str(helices_list[i]))
    return(helices_list, helices_names)

def validate_protein_sequence(seq):
  if re.search(r"[^AVMLIPFWGSCTYQNKRHEDavmlipfwgsctyqnkrhed]", seq): #if the sequence contains anything different than the 20 amino acids
    return False
  else:
    return True


def create_plot(all_seq, all_annotation, output_file_path, label1, label2,label3, label4, colors):

  label_indices = [i for i, x in enumerate(all_annotation) if x == label1]
  x = [all_seq[value] for value in label_indices]
  x.sort()

  label_indices = [i for i, x in enumerate(all_annotation) if x == label2]
  y = [all_seq[value] for value in label_indices]
  y.sort()

  label_indices = [i for i, x in enumerate(all_annotation) if x == label3]
  xx = [all_seq[value] for value in label_indices]
  xx.sort()

  label_indices = [i for i, x in enumerate(all_annotation) if x == label4]
  yy = [all_seq[value] for value in label_indices]
  yy.sort()

  fig = Figure()
  fig.set_size_inches(15, 5)
  axis = fig.add_subplot()

  axis.hist([x, y, xx, yy], bins = np.arange(21)-0.5, rwidth=0.6, color=colors, label=[label1, label2, label3, label4], density = [True, True, True, True])
  axis.legend(loc="upper right")
  axis.set_xlabel("Amino acid")
  axis.set_ylabel("Relative Abundance")

  fig.suptitle("AAs relative distribution per classification")
  fig.savefig(output_file_path)


  #fig.show()
  #return fig 
