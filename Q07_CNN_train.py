# -*- coding: utf-8 -*-
"""
Created on Mon May 14 18:17:22 2018

@author: rafael

Lista 02 - Questão 07~08: Treinamento de uma CNN para identificação de vogais
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import pandas as pd

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu

tf.logging.set_verbosity(tf.logging.INFO)

def eval_confusion_matrix(labels, predictions):
    with tf.variable_scope("eval_confusion_matrix"):
        con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=5)

        con_matrix_sum = tf.Variable(tf.zeros(shape=(5,5), dtype=tf.int32),
                                            trainable=False,
                                            name="confusion_matrix_result",
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])


        update_op = tf.assign_add(con_matrix_sum, con_matrix)

        return tf.convert_to_tensor(con_matrix_sum), update_op

## definindo a rede convolutiva
def cnn_model_fn(features, labels, mode):
  """Função modelo para a CNN."""
  # camada de entrada
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Camada de convolução #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5,5],
      padding="same",
      activation=tf.nn.relu)

  # Camada de pooling #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Camada de convolução #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5,5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # camada densa 1.024 neurônios com uma taxa de regularização de dropout
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Camada de saída dos dígitos: 5 dígitos
  logits = tf.layers.dense(inputs=dropout, units=5)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilidades": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculo da perda (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configurar op. de treino (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Métricas de avaliação (for EVAL mode)
  eval_metric_ops = {"precisão": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]), 
                     "conv_matrix": eval_confusion_matrix(labels, predictions["classes"]),}
                     
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    
    # Load training and eval data
    train_frame = pd.read_csv("data/emnist-vowels-train.csv")
    train_data = np.asarray(train_frame.iloc[:,2:].as_matrix(), dtype=np.float32)

    train_labels = np.asarray(train_frame.iloc[:,1].as_matrix(), dtype=np.int32)
    
    del train_frame
    
    test_frame = pd.read_csv("data/emnist-vowels-test.csv")
    eval_data = np.asarray(test_frame.iloc[:,2:].as_matrix(), dtype=np.float32)
    eval_labels = np.asarray(test_frame.iloc[:,1].as_matrix(), dtype=np.int32)

    del test_frame

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="save/CNN_Q07")
      
    # Set up logging for predictions
    tensors_to_log = {"probabilidades": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000)
        
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])
        
    ## execution time    
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
  tf.app.run()


