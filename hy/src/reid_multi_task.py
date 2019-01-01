import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import math
import keras.backend as K
import numpy as np
import glob
import pandas as pd
import random
from PIL import Image
import sys
import time
from tqdm import tqdm
import pickle
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.generic_utils import Progbar
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, CSVLogger
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, Dense, Flatten,Convolution2D
from keras.layers import Reshape, TimeDistributed, Activation,PReLU
from keras.layers.pooling import GlobalAveragePooling2D,MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.layers import Dropout, merge
from keras.regularizers import l2
import keras.losses as losses
from keras.layers import  add, Multiply, Embedding, Lambda
from keras.utils.vis_utils import plot_model 
from keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model 
from keras.initializers import RandomNormal
sys.path.append('../utils')
from datasets import DataManager
from keras.applications import vgg16
from keras.applications.resnet50 import ResNet50
from datasets import split_data
from data_augmentation_reid import ImageGenerator


#from keras.utils.training_utils import multi_gpu_model





tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True
sess = tf.Session(config=tfconfig)
K.set_session(sess)



def single_task_model(classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax', name='fc8', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(x)  #default glorot_uniform
    net = Model(inputs=[base_model.input], outputs=[x])
    for layer in net.layers:
        layer.trainable = True
    return net



def naive_multi_task_model(market_classes,duke_classes,cuhk_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    x = base_model.output
    x = Flatten(name='flatten')(x)

    #market
    m_x = Dropout(0.5)(x)
    #m_x = Dense(2048, name='fc7_1', activation='relu')(m_x)
    #m_x = Dropout(0.5)(m_x)
    m_x = Dense(market_classes, activation='softmax', name='fc8_1', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(m_x)
    market_net = Model(inputs=[base_model.input], outputs=[m_x])

    #duke
    d_x = Dropout(0.5)(x)
    #d_x = Dense(2048, name='fc7_2', activation='relu')(d_x)
    #d_x = Dropout(0.5)(d_x)
    d_x = Dense(duke_classes, activation='softmax', name='fc8_2', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(d_x)
    duke_net = Model(inputs=[base_model.input], outputs=[d_x])

    #cuhk
    c_x = Dropout(0.5)(x)
    #c_x = Dense(2048, name='fc7_3', activation='relu')(c_x)
    #c_x = Dropout(0.5)(m_x)
    c_x = Dense(cuhk_classes, activation='softmax', name='fc8_3', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(c_x)
    cuhk_net = Model(inputs=[base_model.input], outputs=[c_x])

    return market_net,duke_net,cuhk_net

def alternate_multi_task_model(market_classes,duke_classes,cuhk_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    x = base_model.output
    x = Flatten(name='flatten')(x)

    #market
    m_x = Dropout(0.5)(x)
    #m_x = Dense(2048, name='fc7_1', activation='relu')(m_x)
    #m_x = Dropout(0.5)(m_x)
    m_x = Dense(market_classes, activation='softmax', name='fc8_1', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(m_x)

    #duke
    d_x = Dropout(0.5)(x)
    #d_x = Dense(2048, name='fc7_2', activation='relu')(d_x)
    #d_x = Dropout(0.5)(d_x)
    d_x = Dense(duke_classes, activation='softmax', name='fc8_2', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(d_x)

    #cuhk
    c_x = Dropout(0.5)(x)
    #c_x = Dense(2048, name='fc7_3', activation='relu')(c_x)
    #c_x = Dropout(0.5)(c_x)
    c_x = Dense(cuhk_classes, activation='softmax', name='fc8_3', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(c_x)
    multitask_net = Model(inputs=[base_model.input], outputs=[m_x,d_x,c_x])
    return multitask_net



def write_log(callback,names,logs,batch_no):
    for name,value in zip(names,logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary,batch_no)
        callback.writer.flush()

#define the result path
def path_file(isCenterloss,model_names,task_name):
    if isCenterloss:
        log_path = '../new_logs/' + model_names +'/'+task_name+'/center_loss/'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        stored_model_path = '../new_trained_models/'+model_names+'/'+task_name+'/center_loss/'
        if not os.path.exists(stored_model_path):
            os.makedirs(stored_model_path)
    else:
        log_path = '../new_logs/' + model_names +'/'+task_name+'/softmax_loss/'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        stored_model_path =  '../new_trained_models/'+model_names+'/'+task_name+'/softmax_loss/'
        if not os.path.exists(stored_model_path):
            os.makedirs(stored_model_path)
    return log_path,stored_model_path


def generator(dataset_name,batch_size,classes):
    data_loader = DataManager(dataset_name)
    ground_truth_data = data_loader.get_data()
    images_path = data_loader.dataset_path
    train_keys,val_keys = split_data(ground_truth_data,0)
    image_generator = ImageGenerator(ground_truth_data, batch_size,
                                     [224,224,3],
                                     train_keys, 
                                     val_keys, 
                                     classes,
                     None,
                     path_prefix=images_path,
                                     grayscale = False)
    train_generator = image_generator.flow(mode='train')
    train_num = len(train_keys)/batch_size
    return train_generator,train_num

def run_model(is_single,is_naive,is_alternative,lr,loss_weights_1,loss_weights_2,loss_weights_3,dataset_name_1,dataset_name_2,dataset_name_3,batch_size,epoch,stored_model_path,log_path):
    train_generator_1,train_num_1 = generator(dataset_name_1,batch_size,market_classes)
    train_generator_2,train_num_2 = generator(dataset_name_2,batch_size,duke_classes)
    train_generator_3,train_num_3 = generator(dataset_name_3,batch_size,cuhk_classes)
    print train_num_1,train_num_2,train_num_3

    single_market = single_task_model(751)
    single_duke = single_task_model(702)
    single_cuhk = single_task_model(743)

    naive_market,naive_duke,naive_cuhk = naive_multi_task_model(751,702,743)

    alternate_1 = alternate_multi_task_model(751,702,743)
    alternate_2 = alternate_multi_task_model(751,702,743)
    alternate_3 = alternate_multi_task_model(751,702,743)

    single_market.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    single_duke.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    single_cuhk.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    naive_market.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    naive_duke.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    naive_cuhk.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    alternate_1.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy', loss_weights = [loss_weights_1,1,1],metrics=['accuracy'])
    alternate_2.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy', loss_weights = [1,loss_weights_2,1],metrics=['accuracy'])
    alternate_3.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy', loss_weights = [1,1,loss_weights_3],metrics=['accuracy'])
    
    single_market.summary()
    naive_market.summary()
    alternate_1.summary()

    callback = TensorBoard(log_path)
    callback.set_model(alternate_1)
    LOSSES_11 = []
    LOSSES_12 = []
    LOSSES_13 = []
    LOSSES_21 = []
    LOSSES_22 = []
    LOSSES_23 = []
    LOSSES_31 = []
    LOSSES_32 = []
    LOSSES_33 = []

    LOSSES_test_11 = []
    LOSSES_test_12 = []
    LOSSES_test_13 = []
    LOSSES_test_21 = []
    LOSSES_test_22 = []
    LOSSES_test_23 = []
    LOSSES_test_31 = []
    LOSSES_test_32 = []
    LOSSES_test_33 = []


    EPOCH_NB = epoch
    BATCH_NB_1 = train_num_1
    BATCH_NB_2 = train_num_2
    BATCH_NB_3 = train_num_3

    single_market_accuracy = 0
    single_duke_accuracy =  0
    single_cuhk_accuracy = 0

    naive_market_accuracy = 0
    naive_duke_accuracy = 0
    naive_cuhk_accuracy = 0

    alternate_market_accuracy = 0
    alternate_duke_accuracy = 0
    alternate_cuhk_accuracy = 0

    for epoch_nb in xrange(EPOCH_NB):
        Losses_11=[]
        Losses_12=[]
        Losses_13=[]

        Losses_21=[]
        Losses_22=[]
        Losses_23=[]

        Losses_31=[]
        Losses_32=[]
        Losses_33=[]

        if epoch_nb%3==0:
            for batch_nb in tqdm(xrange(BATCH_NB_1), total=BATCH_NB_1):
                [Image_data_1, Labels_1] = train_generator_1.next()
                if is_single:
                    losses_single_market = single_market.train_on_batch(Image_data_1['input'],Labels_1['label'])
                    write_log(callback,single_market_names,losses_single_market,batch_nb)
                    Losses_11.append(losses_single_market)
                if is_naive:
                    losses_naive_market = naive_market.train_on_batch(Image_data_1['input'],Labels_1['label'])
                    write_log(callback,naive_market_names,losses_naive_market,batch_nb)
                    Losses_12.append(losses_naive_market)
                if is_alternative:
                    losses_alternate_market = alternate_1.train_on_batch(Image_data_1['input'],
                        [Labels_1['label'],alternate_2.predict(Image_data_1['input'])[1],alternate_3.predict(Image_data_1['input'])[2]])
                    write_log(callback,alternate_market_names,losses_alternate_market,batch_nb)
                    Losses_13.append(losses_alternate_market)

            if is_single:
                if np.array(Losses_11).mean(axis=0)[1]>single_market_accuracy:
                    single_market_accuracy = np.array(Losses_11).mean(axis=0)[1]
                    single_market.save(stored_model_path+str(epoch_nb)+'_'+'single_market'+'_'+str(single_market_accuracy)+'__'+'weights.h5')
                print 'Done for Epoch %d.'% epoch_nb
                print  'single_market_train_result:',np.array(Losses_11).mean(axis=0)
            if is_naive:
                if np.array(Losses_12).mean(axis=0)[1]>naive_market_accuracy:
                    naive_market_accuracy = np.array(Losses_12).mean(axis=0)[1]
                    naive_market.save(stored_model_path+str(epoch_nb)+'_'+'naive_market'+'_'+str(naive_market_accuracy)+'__'+'weights.h5')
                print 'Done for Epoch %d.'% epoch_nb
                print  'naive_market_train_result:',np.array(Losses_12).mean(axis=0)
            if is_alternative:
                if np.array(Losses_13).mean(axis=0)[4]>alternate_market_accuracy:
                    alternate_market_accuracy = np.array(Losses_13).mean(axis=0)[4]
                    alternate_1.save(stored_model_path+str(epoch_nb)+'_'+'alternate_market'+'_'+str(alternate_market_accuracy)+'__'+'weights.h5')
                print 'Done for Epoch %d.'% epoch_nb
                print  'alternate_market_train_result:',np.array(Losses_13).mean(axis=0)
                   
        elif epoch_nb%3==1:
            for batch_nb in tqdm(xrange(BATCH_NB_2), total=BATCH_NB_2):
                [Image_data_2, Labels_2] = train_generator_2.next()
                if is_single:
                    losses_single_duke = single_duke.train_on_batch(Image_data_2['input'],Labels_2['label'])
                    write_log(callback,single_duke_names,losses_single_duke,batch_nb)
                    Losses_21.append(losses_single_duke)              
                if is_naive:
                    losses_naive_duke = naive_duke.train_on_batch(Image_data_2['input'],Labels_2['label'])
                    write_log(callback,naive_duke_names,losses_naive_duke,batch_nb)
                    Losses_22.append(losses_naive_duke)
                if is_alternative:
                    losses_alternate_duke = alternate_2.train_on_batch(Image_data_2['input'],
                        [alternate_1.predict(Image_data_2['input'])[0],Labels_2['label'],alternate_3.predict(Image_data_2['input'])[2]])
                    write_log(callback,alternate_duke_names,losses_alternate_duke,batch_nb)
                    Losses_23.append(losses_alternate_duke)               
            if is_single:
                if np.array(Losses_21).mean(axis=0)[1]>single_duke_accuracy:
                    single_duke_accuracy = np.array(Losses_21).mean(axis=0)[1]
                    single_duke.save(stored_model_path+str(epoch_nb)+'_'+'single_duke'+'_'+str(single_duke_accuracy)+'__'+'weights.h5')
                print 'Done for Epoch %d.'% epoch_nb
                print  'single_duke_train_result:',np.array(Losses_21).mean(axis=0)
            if is_naive:
                if np.array(Losses_22).mean(axis=0)[1]>naive_duke_accuracy:
                    naive_duke_accuracy = np.array(Losses_22).mean(axis=0)[1]
                    naive_market.save(stored_model_path+str(epoch_nb)+'_'+'naive_market'+'_'+str(naive_duke_accuracy)+'__'+'weights.h5')
                print 'Done for Epoch %d.'% epoch_nb
                print  'naive_duke_train_result:',np.array(Losses_22).mean(axis=0)
            if is_alternative:
                if np.array(Losses_23).mean(axis=0)[5]>alternate_duke_accuracy:
                    alternate_duke_accuracy = np.array(Losses_23).mean(axis=0)[5]
                    alternate_2.save(stored_model_path+str(epoch_nb)+'_'+'alternate_duke'+'_'+str(alternate_duke_accuracy)+'__'+'weights.h5')
                print 'Done for Epoch %d.'% epoch_nb
                print  'alternate_duke_train_result:',np.array(Losses_23).mean(axis=0)
            
        else:
            for batch_nb in tqdm(xrange(BATCH_NB_3), total=BATCH_NB_3):
                [Image_data_3, Labels_3] = train_generator_3.next()
                if is_single:
                    losses_single_cuhk = single_cuhk.train_on_batch(Image_data_3['input'],Labels_3['label'])
                    write_log(callback,single_cuhk_names,losses_single_cuhk,batch_nb)
                    Losses_31.append(losses_single_cuhk)
                if is_naive:
                    losses_naive_cuhk = naive_cuhk.train_on_batch(Image_data_3['input'],Labels_3['label'])
                    write_log(callback,naive_cuhk_names,losses_naive_cuhk,batch_nb)
                    Losses_32.append(losses_naive_cuhk)
                if is_alternative:
                    losses_alternate_cuhk = alternate_3.train_on_batch(Image_data_3['input'],
                        [alternate_1.predict(Image_data_3['input'])[0],alternate_2.predict(Image_data_3['input'])[1],Labels_3['label'],])
                    write_log(callback,alternate_cuhk_names,losses_alternate_cuhk,batch_nb)          
                    Losses_33.append(losses_alternate_cuhk)

            if is_single:
                if np.array(Losses_31).mean(axis=0)[1]>single_cuhk_accuracy:
                    single_cuhk_accuracy = np.array(Losses_31).mean(axis=0)[1]
                    single_cuhk.save(stored_model_path+str(epoch_nb)+'_'+'single_cuhk'+'_'+str(single_cuhk_accuracy)+'__'+'weights.h5')
                print 'Done for Epoch %d.'% epoch_nb
                print  'single_cuhk_train_result:',np.array(Losses_31).mean(axis=0)
            if is_naive:
                if np.array(Losses_32).mean(axis=0)[1]>naive_market_accuracy:
                    naive_cuhk_accuracy = np.array(Losses_32).mean(axis=0)[1]
                    naive_cuhk.save(stored_model_path+str(epoch_nb)+'_'+'naive_cuhk'+'_'+str(naive_cuhk_accuracy)+'__'+'weights.h5')
                print 'Done for Epoch %d.'% epoch_nb
                print  'naive_cuhk_train_result:',np.array(Losses_32).mean(axis=0)
            if is_alternative:
                if np.array(Losses_33).mean(axis=0)[6]>alternate_market_accuracy:
                    alternate_cuhk_accuracy = np.array(Losses_33).mean(axis=0)[6]
                    alternate_3.save(stored_model_path+str(epoch_nb)+'_'+'alternate_cuhk'+'_'+str(alternate_cuhk_accuracy)+'__'+'weights.h5')
                print 'Done for Epoch %d.'% epoch_nb
                print  'alternate_cuhk_train_result:',np.array(Losses_33).mean(axis=0)


if __name__ == '__main__':
    dataset_name_1 = 'market'
    dataset_name_2='duke'
    dataset_name_3 = 'cuhk'
    task_name = 'reid'
    model_names = 'ResNet50'
    isCenterloss = False
    loss_weights_1=2
    loss_weights_2=2
    loss_weights_3=2
    market_classes = 751
    duke_classes = 702
    cuhk_classes = 743
    epoch = 512
    batch_size = 32
    lr = 0.01
    is_single=True
    is_naive = False
    is_alternative = False

    single_market_names = ['single_market_loss','single_market_acc']
    single_duke_names = ['single_duke_loss','single_duke_acc']
    single_cuhk_names = ['single_cuhk_loss','single_cuhk_acc']

    naive_market_names = ['naive_market_loss','naive_market_acc']
    naive_duke_names = ['naive_duke_loss','naive_duke_acc']
    naive_cuhk_names = ['naive_cuhk_loss','naive_cuhk_acc']

    alternate_market_names = ['alternate_1_total_loss','market_loss_1','duke_loss_1','cuhk_loss_1','market_acc_1','duke_acc_1','cuhk_acc_1']
    alternate_duke_names = ['alternate_2_total_loss','market_loss_2','duke_loss_2','cuhk_loss_2','market_acc_2','duke_acc_2','cuhk_acc_2']
    alternate_cuhk_names = ['alternate_3_total_loss','market_loss_3','duke_loss_3','cuhk_loss_3','market_acc_3','duke_acc_3','cuhk_acc_3']

    
    log_path,stored_model_path=path_file(isCenterloss,model_names,task_name)
    run_model(is_single,is_naive,is_alternative,lr,loss_weights_1,loss_weights_2,loss_weights_3,dataset_name_1,dataset_name_2,dataset_name_3,batch_size,epoch,stored_model_path,log_path)

