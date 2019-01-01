from __future__ import division, print_function, absolute_import

import os
import sys
import numpy as np
import tensorflow as tf
#from keras.applications.resnet50 import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
sys.path.append('../utils')
from preprocessor import preprocess_input
from file_helper import write
import scipy.io as sio
import matplotlib.pyplot as plt


def extract_info(dir_path):
    infos = []
    for image_name in sorted(os.listdir(dir_path)):
        if '.txt' in image_name:
            continue
        if 's' in image_name or 'f' in image_name:
            # market && duke
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1][1])
        elif 's' not in image_name:
            # grid
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1])
        else:
            continue
        infos.append((person, camera))

    return infos


def extract_feature(dir_path, net):
    features = []
    person_id = []
    infos=[]
    cam_id = []
    for image_name in sorted(os.listdir(dir_path)):
        if '.txt' in image_name:
            continue
        if 'f' in image_name or 's' in image_name:
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1][1])
        elif 's' not in image_name:
            # grid
            print (image_name)
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1])
        else:
            continue
        image_path = os.path.join(dir_path, image_name)
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = net.predict(x)
        features.append(np.squeeze(feature))
        person_id.append(person)
        cam_id.append(camera)
        infos.append((person, camera))

    return features, infos,person_id,cam_id


def similarity_matrix(query_f, test_f):
    # Tensorflow graph
    # use GPU to calculate the similarity matrix
    query_t = tf.placeholder(tf.float32, (None, None))
    test_t = tf.placeholder(tf.float32, (None, None))
    query_t_norm = tf.nn.l2_normalize(query_t, dim=1)
    test_t_norm = tf.nn.l2_normalize(test_t, dim=1)
    tensor = tf.matmul(query_t_norm, test_t_norm, transpose_a=False, transpose_b=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    result = sess.run(tensor, {query_t: query_f, test_t: test_f})
    print(result.shape)
    # descend
    return result


def sort_similarity(query_f, test_f):
    result = similarity_matrix(query_f, test_f)
    result_argsort = np.argsort(-result, axis=1)
    return result, result_argsort


def map_rank_quick_eval(query_info, test_info, result_argsort):
    # much more faster than hehefan's evaluation
    match = []
    junk = []
    QUERY_NUM = len(query_info)

    #[3368,19732]
    #print ('query matrix is:',result_argsort.shape)
    #print('test_length:',len(test_info))[19732]
    #print('query_length:',len(query_info))[3368]
    #print ('test_info:',test_info)
    #print('query_info:',query_info)
    for q_index, (qp, qc) in enumerate(query_info):
        tmp_match = []
        tmp_junk = []
        for t_index in range(len(test_info)):
            p_t_idx = result_argsort[q_index][t_index]
            p_info = test_info[int(p_t_idx)]

            tp = p_info[0]
            tc = p_info[1]
            if tp == qp and qc != tc:
                tmp_match.append(t_index)
            elif tp == qp or tp == -1:
                tmp_junk.append(t_index)
        match.append(tmp_match)
        junk.append(tmp_junk)

    rank_1 = 0.0
    mAP = 0.0
    rank1_list = list()
    for idx in range(len(query_info)):
        if idx % 100 == 0:
            print('evaluate img %d' % idx)
        recall = 0.0
        precision = 1.0
        ap = 0.0
        YES = match[idx]
        IGNORE = junk[idx]
        ig_cnt = 0
        for ig in IGNORE:
            if ig < YES[0]:
                ig_cnt += 1
            else:
                break
        if ig_cnt >= YES[0]:
            rank_1 += 1
            rank1_list.append(1)
        else:
            rank1_list.append(0)

        for i, k in enumerate(YES):
            ig_cnt = 0
            for ig in IGNORE:
                if ig < k:
                    ig_cnt += 1
                else:
                    break
            cnt = k + 1 - ig_cnt
            hit = i + 1
            tmp_recall = hit / len(YES)
            tmp_precision = hit / cnt
            ap = ap + (tmp_recall - recall) * ((precision + tmp_precision) / 2)
            recall = tmp_recall
            precision = tmp_precision

        mAP += ap
    rank1_acc = rank_1 / QUERY_NUM
    mAP = mAP / QUERY_NUM
    print('Rank 1:\t%f' % rank1_acc)
    print('mAP:\t%f' % mAP)
    np.savetxt('rank_1.log', np.array(rank1_list), fmt='%d')
    return rank1_acc, mAP


def train_predict(net, train_path, pid_path, score_path):
    net = Model(inputs=[net.input], outputs=[net.get_layer('avg_pool').output])
    train_f, test_info = extract_feature(train_path, net)
    result, result_argsort = sort_similarity(train_f, train_f)
    for i in range(len(result)):
        result[i] = result[i][result_argsort[i]]
    result = np.array(result)
    # ignore top1 because it's the origin image
    np.savetxt(score_path, result[:, 1:], fmt='%.4f')
    np.savetxt(pid_path, result_argsort[:, 1:], fmt='%d')
    return result


def test_predict(net, probe_path, gallery_path, pid_path, score_path):
    net = Model(inputs=[net.input], outputs=[net.get_layer('avg_pool').output])
    #net = Model(inputs=[net.input], outputs=[net.get_layer('fc8').output])
    test_f, test_info,test_person_id,test_cam_id = extract_feature(gallery_path, net)
    query_f, query_info,query_person_id,query_cam_id = extract_feature(probe_path, net)
    result, result_argsort = sort_similarity(query_f, test_f)
    for i in range(len(result)):
        result[i] = result[i][result_argsort[i]]
    result = np.array(result)
    np.savetxt(pid_path, result_argsort, fmt='%d')
    np.savetxt(score_path, result, fmt='%.4f')


def result_eval(predict_path, log_path, TEST, QUERY):
    res = np.genfromtxt(predict_path, delimiter=' ')
    print('predict info get, extract gallery info start')
    test_info = extract_info(TEST)
    print('extract probe info start')
    query_info = extract_info(QUERY)
    print('start evaluate map and rank acc')
    rank1, mAP = map_rank_quick_eval(query_info, test_info, res)
    write(log_path, predict_path + '\n')
    write(log_path, '%f\t%f\n' % (rank1, mAP))


def grid_result_eval(predict_path, log_path='grid_eval.log'):
    pids4probes = np.genfromtxt(predict_path, delimiter=' ')
    probe_shoot = [0, 0, 0, 0, 0]
    for i, pids in enumerate(pids4probes):
        for j, pid in enumerate(pids):
            if pid - i == 775:
                if j == 0:
                    for k in range(5):
                        probe_shoot[k] += 1
                elif j < 5:
                    for k in range(1, 5):
                        probe_shoot[k] += 1
                elif j < 10:
                    for k in range(2, 5):
                        probe_shoot[k] += 1
                elif j < 20:
                    for k in range(3, 5):
                        probe_shoot[k] += 1
                elif j < 50:
                    for k in range(4, 5):
                        probe_shoot[k] += 1
                break
    probe_acc = [shoot / len(pids4probes) for shoot in probe_shoot]
    write(log_path, predict_path + '\n')
    write(log_path, '%.2f\t%.2f\t%.2f\n' % (probe_acc[0], probe_acc[1], probe_acc[2]))
    print(predict_path)
    print(probe_acc)

def evaluation_result(dataset_name,mode,model_path):
    net = load_model(model_path,compile=False)
    #net = Model(inputs=[net.input], outputs=[net.get_layer('avg_pool').output])
    net = Model(inputs=[net.input], outputs=[net.get_layer('flatten').output])
    if dataset_name=='market':
        probe_path='../data/Market-1501/query'
        gallery_path = '../data/Market-1501/bounding_box_test'
    elif dataset_name=='duke':
        gallery_path = '../data/DukeMTMC-reID/bounding_box_test'
        probe_path = '../data/DukeMTMC-reID/query'
    else:
        print('no such database')
   
    pid_path =  '../result/' + mode + '_'  + dataset_name+ '_feature.txt' 
    score_path =  '../result/'+mode + '_'  + dataset_name+ '_score.txt' 
    log_path = '../result/'  + mode + '_'  + dataset_name+ '_rank1_map.log' 
    if os.path.exists(pid_path) == False:
        test_f, test_info,test_person_id,test_cam_id = extract_feature(gallery_path, net)
        query_f, query_info,query_person_id,query_cam_id = extract_feature(probe_path, net)
        save_path = '../result/'+dataset_name+'_'+mode+'_'+'result.mat'
        sio.savemat(save_path,{'query_f':query_f,'query_label':query_person_id,'query_cam':query_cam_id,'gallery_f':test_f,'gallery_label':test_person_id,'gallery_cam':test_cam_id})
        result, result_argsort = sort_similarity(query_f, test_f)
        for i in range(len(result)):
            result[i] = result[i][result_argsort[i]]
        result = np.array(result)
        np.savetxt(pid_path, result_argsort, fmt='%d')
        np.savetxt(score_path, result, fmt='%.4f')
    result_eval(pid_path, log_path, gallery_path, probe_path)




if __name__ == '__main__':
    #model_path = '../new_trained_models/ResNet50/reid/softmax_loss/alternate_market_0.822324__weights.h5'
    #model_path = '../new_trained_models/ResNet50/reid/softmax_loss/alternate_market_0.00595606__weights.h5'
    model_path = '../new_trained_models/ResNet50/reid/softmax_loss/231_single_market_0.884901__weights.h5'
    #model_path = '../new_trained_models/ResNet50/reid/softmax_loss/single_market_0.855739__weights.h5'
    #model_path = '../new_trained_models/ResNet50/reid/softmax_loss/single_duke_0.871487__weights.h5'
    #model_path = '../new_trained_models/ResNet50/reid/softmax_loss/naive_market_0.843982__weights.h5'
    #model_path = '../new_trained_models/ResNet50/reid/softmax_loss/naive_duke_0.879845__weights.h5'
    dataset_name = 'market'
    mode = 'single'
    evaluation_result(dataset_name,mode,model_path)


