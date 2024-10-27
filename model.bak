import os
import cv2
import math
import json
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image as im
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation
from keras.regularizers import l2
from keras.models import Model
from mmpose.apis.inferencers import MMPoseInferencer

CLEANER = ' ' * 30 + '\n'
LABELMAP = {0: 'A01',1: 'A02',2: 'A03',3: 'A04',4: 'A05',5: 'A06',6: 'A07',7: 'A08',8: 'A17',9: 'A18',10: 'A19',11: 'A20',12: 'A21',13: 'A22',14: 'A23',15: 'A24',16: 'A25',17: 'A26',18: 'A30',19: 'A31',
}

def init_args(video=None):
    if video is None:
        video = './static/demo/clip_short.mp4'
    args = {
        'video': video,
        'c3d_checkpoint': './checkpoint/c3d_checkpoint.h5',
        'seq_checkpoint': './checkpoint/seq_checkpoint.h5',
        'resnet_checkpoint': './checkpoint/resnet_checkpoint.pth',
        'scaler': './checkpoint/scaler.pkl',
        'svm': './checkpoint/svm.pkl',
        'learning_rate': 0.005,
        'momentum': 0.9,
        'inference_frame_num': 16,
        'resize': (171, 128),
        'max_seq_length': 20,
        'num_feature': 2048,
        'max_image_size': (320, 180),
        'rescale': 6,
        'limit_frame': 256,
        'extract_args': {
            'init_args': {
                'pose2d': 'rtmo', 
                'pose2d_weights': "./checkpoint/rtmo_checkpoint.pth", 
                'scope': 'mmpose', 
                'device': 'cuda:0', 
                'det_model': None, 
                'det_weights': None, 
                'det_cat_ids': 0, 
                'pose3d': None, 
                'pose3d_weights': None, 
                'show_progress': False
            },
            'call_args': {
                'inputs': video, 
                'show': False, 
                'draw_bbox': True, 
                'draw_heatmap': False, 
                'bbox_thr': 0.5, 
                'nms_thr': 0.65, 
                'pose_based_nms': True, 
                'kpt_thr': 0.3, 
                'tracking_thr': 0.3, 
                'use_oks_tracking': False, 
                'disable_norm_pose_2d': False, 
                'disable_rebase_keypoint': False, 
                'num_instances': 1, 
                'radius': 3, 
                'thickness': 1, 
                'skeleton_style': 'openpose', 
                'black_background': False, 
                'vis_out_dir': '', 
                'pred_out_dir': '', 
                'vis-out-dir': './'
            }
        }
    }
    return args

def init_c3d_model(args):
    input_shape = (112,112,16,3)
    weight_decay = 0.005
    nb_classes = 20

    inputs = Input(input_shape)
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')(x)
    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)
    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)
    x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)
    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(x)
    x = Activation('softmax')(x)

    model = Model(inputs, x)
    sgd = SGD(learning_rate=args['learning_rate'], momentum=args['momentum'], nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model = tf.keras.models.load_model(args['c3d_checkpoint'])
    return model

def init_seq_model(args):
    frame_features_input = keras.Input((args['max_seq_length'], args['num_feature']))
    mask_input = keras.Input((args['max_seq_length'],), dtype="bool")

    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(21, activation="softmax")(x)
    
    model = keras.Model([frame_features_input, mask_input], output)
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.load_weights(args['seq_checkpoint'])
    return model

def init_feature_extractor(args):
    feature_extractor = keras.applications.ResNet50(
        # weights="imagenet",
        weights=args['resnet_checkpoint'],
        include_top=False,
        pooling="avg",
        input_shape=(args['max_image_size'][1], args['max_image_size'][0], 3),
    )
    preprocess_input = keras.applications.resnet.preprocess_input

    inputs = keras.Input((args['max_image_size'][1], args['max_image_size'][0], 3)) 
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

def marking_bold_point(args, target, x, y):
    if x<0 or y <0 or x >= args['max_image_size'][0] or y >= args['max_image_size'][1]:
        return 
    if y-1 >= 0 and x-1 >= 0 :
        target[args['max_image_size'][0] * (y-1) + (x-1)] = 255
    if y-1 >= 0 :
        target[args['max_image_size'][0] * (y-1) + (x)] = 255
    if y-1 >= 0 and x+1 < args['max_image_size'][0]:    
        target[args['max_image_size'][0] * (y-1) + (x+1)] = 255
    if x-1 >= 0 :
        target[args['max_image_size'][0] * (y) + (x-1)] = 255
    target[args['max_image_size'][0] * (y) + (x)] = 255
    if x+1 < args['max_image_size'][0]:
        target[args['max_image_size'][0] * (y) + (x+1)] = 255
    if y+1 < args['max_image_size'][1] and x-1 >= 0 :
        target[args['max_image_size'][0] * (y+1) + (x-1)] = 255
    if y+1 < args['max_image_size'][1]:
        target[args['max_image_size'][0] * (y+1) + (x)] = 255    
    if y+1 < args['max_image_size'][1] and  x+1 < args['max_image_size'][0]:
        target[args['max_image_size'][0] * (y+1) + (x+1)] = 255
    return target

def transform_bitmap(args, point1, point2):
    target = [0] * (args['max_image_size'][0] * args['max_image_size'][1])
    # TBD
    for i in range(len(point1)):
        marking_bold_point(args, target, point1[i], point2[i])
    target = np.reshape(np.fromiter(target, dtype=np.uint8), args['max_image_size'])
    data = im.fromarray(target)
    data = data.convert('RGB')
    return data

def pre_processing(keypoints):
    processed_keypoints = [0 for _ in range(15)]
    processed_keypoints[0] = keypoints[0]
    processed_keypoints[1] = [(keypoints[5][0] + keypoints[6][0])/2, (keypoints[5][1] + keypoints[6][1])/2]
    processed_keypoints[2] = keypoints[6]
    processed_keypoints[3] = keypoints[8]
    processed_keypoints[4] = keypoints[10]
    processed_keypoints[5] = keypoints[5]
    processed_keypoints[6] = keypoints[7]
    processed_keypoints[7] = keypoints[9]
    processed_keypoints[8] = [(keypoints[12][0] + keypoints[11][0])/2, (keypoints[12][1] + keypoints[11][1])/2]
    processed_keypoints[9] = keypoints[12]
    processed_keypoints[10] = keypoints[14]
    processed_keypoints[11] = keypoints[16]
    processed_keypoints[12] = keypoints[11]
    processed_keypoints[13] = keypoints[13]
    processed_keypoints[14] = keypoints[15]
    return processed_keypoints

def extract_video(args):
    print('\033[1;32minference Action...\033[0m', end='\r', flush=True)
    init_args = args['init_args']
    call_args = args['call_args']
    file_data = []
    inferencer = MMPoseInferencer(**init_args)
    cap = cv2.VideoCapture(call_args['inputs'])
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                print(f'\033[1;32mExtracting Skeleton... ({frame_index}/{frame_count})\033[0m', end='\r', flush=True)
                temp_call_args = call_args
                temp_call_args['inputs'] = frame
                results = inferencer(**temp_call_args)
                frame_data = {
                    'frame_index': frame_index,
                    'objects': [],
                    'persons': []
                }

                for index, result in enumerate(results):
                    person_data = {
                        'index': index,
                        'person_index': index,
                    }
                    pred = result['predictions'][0]
                    pred.sort(key = lambda x: x['bbox'][0][0])
                    for p in pred:
                        x1, y1, x2, y2 = p['bbox'][0]
                        person_data['person_center'] = [(x1 + x2)/2, (y1 + y2)/2]
                        person_data['keypoints'] = pre_processing(p['keypoints'])
                    frame_data['persons'].append(person_data)
                file_data.append(frame_data)
                frame_index += 1
            else:
                break
    cap.release()
    print('\033[1;32mExtracting... Done!\033[0m', end=CLEANER)
    return file_data

def extract_skeleton(args, frames):
    empty_list_x=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    empty_list_y=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    bitmap_frames= []

    for frame_num in range(args['limit_frame']):
        frame = frames[frame_num]
        point1 = []
        point2 = []
        point3 = []
        point4 = []
        person_length = len(frame['persons'])
        if person_length ==0:
            data = transform_bitmap(args, empty_list_x+empty_list_x,empty_list_y+empty_list_y)
        elif person_length==1:
            a=frame['persons'][0]['keypoints']
            point1 = [int(float(point[0])//args['rescale']) for point in a]
            point2 = [int(float(point[1])//args['rescale']) for point in a]
            data = transform_bitmap(args, point1+empty_list_x , point2+empty_list_y)
        else:
            a=frame['persons'][0]['keypoints']
            b=frame['persons'][1]['keypoints'] 

            point1 = [int(float(point[0])//args['rescale']) for point in a]
            point2 = [int(float(point[1])//args['rescale']) for point in a]
            point3 = [int(float(point[0])//args['rescale']) for point in b]
            point4 = [int(float(point[1])//args['rescale']) for point in b]

            data = transform_bitmap(args, point1+point3,point2+point4)     
        bitmap_frames.append(data)     
        
    return bitmap_frames

def extract_feature_mask(args, feature_extractor, frames):
    max_width, max_height = args['max_image_size']
    processed_frames = []

    for frame in frames:
        frame_array = np.array(frame)
        if frame_array.shape[0] > max_height or frame_array.shape[1] > max_width:
            frame_array = cv2.resize(frame_array, (max_width, max_height))
        padded_frame = np.pad(frame_array, ((0, max_height - frame_array.shape[0]), (0, max_width - frame_array.shape[1]), (0, 0)), mode='constant')
        processed_frames.append(padded_frame)

    frames = np.array(processed_frames)[None, ...]
    mask = np.zeros(shape=(1, args['max_seq_length'],), dtype="bool")
    feature = np.zeros(shape=(1, args['max_seq_length'], args['num_feature']), dtype="float32")
   
    for frame_index, batch in enumerate(frames):
        video_length = batch.shape[1]
        length = min(args['max_seq_length'], video_length)
        for seq_index in range(length):
            feature[frame_index, seq_index, :] = feature_extractor.predict(batch[None, seq_index, :])
        mask[frame_index, :length] = 1
    return feature, mask

def inference_c3d(model, clip):
    inputs = np.array(clip).astype(np.float32)
    inputs = np.expand_dims(inputs, axis=0)
    inputs[..., 0] -= 99.9
    inputs[..., 1] -= 92.1
    inputs[..., 2] -= 82.6
    inputs[..., 0] /= 65.8
    inputs[..., 1] /= 62.3
    inputs[..., 2] /= 60.3
    inputs = inputs[:,:,8:120,30:142,:]
    inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
    pred = model.predict(inputs)
    pred_index = np.argmax(pred[0])
    pred_label = LABELMAP[pred_index]
    return pred_label

def inference_seq(model, vocabulary, freature, mask):
    probabilities = model.predict([freature, mask])[0]
    sorted_indices = np.argsort(probabilities)[::-1]
    # return int(vocabulary[sorted_indices[0]]) - 1
    return int(vocabulary[sorted_indices[0]])

def c3d_module(args):
    print('\033[1;32mInitialize C3D Model...\033[0m', end='\r', flush=True)
    c3d_model = init_c3d_model(args)
    print('\033[1;32mInitialize C3D Model... Done!\033[0m', end=CLEANER)

    clip = []
    cap = cv2.VideoCapture(args['video'])
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cur_frame = 1
    while True:
        flag, frame = cap.read()
        if flag:
            print(f'\033[1;32mPreprocessing... (current frame: {cur_frame}/{num_frame})\033[0m', end='\r', flush=True)
            resorted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(resorted_frame, args['resize'])
            clip.append(resized_frame)
            cur_frame += 1
        else:
            print('\033[1;32mPreprocessing... Done!\033[0m', end=CLEANER)
            break
    
    print('\033[1;32minference Action...\033[0m', end='\r', flush=True)
    inference_result_list = []
    last_idx = min(args['limit_frame'], num_frame - 15)
    for start_idx in range(0, num_frame-15):
        print(f'\033[1;32minference Action... ({start_idx}/{last_idx})\033[0m', end='\r', flush=True)
        end_idx = start_idx + args['inference_frame_num']
        if end_idx < num_frame and start_idx < last_idx:
            label = inference_c3d(c3d_model, clip[start_idx:end_idx])
            inference_result_list.append(label)
        else:
            break
    print('\033[1;32minference Action... Done!\033[0m', end=CLEANER)
    most_common_label = max(set(inference_result_list), key=inference_result_list.count)
    return most_common_label

def seq_module(args):
    print('\033[1;32mInitialize Seq Model...\033[0m', end='\r', flush=True)
    seq_model = init_seq_model(args)
    print('\033[1;32mInitialize Seq Model... Done!\033[0m', end=CLEANER)
    extracted_frames = extract_video(args['extract_args'])
    bitmap_frames = extract_skeleton(args, extracted_frames)
    print('\033[1;32mInitialize Feature Extractor...\033[0m', end='\r', flush=True)
    feature_extractor = init_feature_extractor(args)
    print('\033[1;32mInitialize Feature Extractor... Done!\033[0m', end=CLEANER)
    
    label_processor = tf.keras.layers.experimental.preprocessing.StringLookup(num_oov_indices=0, vocabulary=[str(key) for key in LABELMAP.keys()])
    vocabulary = label_processor.get_vocabulary()
    print('\033[1;32minference Action...\033[0m', end='\r', flush=True)
    feature, mask = extract_feature_mask(args, feature_extractor, bitmap_frames)
    result = inference_seq(seq_model, vocabulary, feature, mask)
    print('\033[1;32minference Action...Done\033[0m', end=CLEANER)
    return result

def categorization(action):
    action_map = {
        0: 0, 4: 0, 'A01': 0, 'A05': 0,
        1: 1, 2: 1, 3: 1, 5: 1, 6: 1, 7: 1, 'A02': 1, 'A03': 1, 'A04': 1, 'A06': 1, 'A07': 1, 'A08': 1,
        8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 'A17': 2, 'A18': 2, 'A19': 2, 'A20': 2, 'A21': 2,
        13: 3, 14: 3, 'A22': 3, 'A23': 3,
        15: 4, 16: 4, 17: 4, 'A24': 4, 'A25': 4, 'A26': 4,
        18: 5, 'A30': 5,
        19: 6, 'A31': 6
    }
    return action_map.get(action, -1)

def combine(args, c3d_category, seq_category):
    top_5 = {'아동방임-방임(C011)':0,'아동학대-신체학대(C012)':0,'주거침임-문(C021)':0,'폭행/강도-흉기(C031)':0, '폭행강도-위협행동(C032)':0,'절도-문앞(C041)':0,'절도-주차장(C042)':0}
    input_data = np.array([[c3d_category, seq_category]])
    normalizer = joblib.load(args['scaler'])
    classifier = joblib.load(args['svm'])
    normalized_data = normalizer.transform(input_data)
    probability = classifier.predict_proba(normalized_data)

    for n, key in enumerate(top_5):
        top_5[key] = round(probability[0][n],6)
    result = sorted(top_5.items(),reverse=True, key =lambda item:item[1])
    return result

async def inference(video=None):
    # INITIALIZING
    yield f"Inference Start\n"
    yield f"Initalizing Arguments...\n"
    args = init_args(video)
    yield f"Initalizing Arguments... Done!\n"

    # C3D MODEL
    yield f"Initialize C3D Model...\n"
    c3d_model = init_c3d_model(args)
    yield f"Initialize C3D Model... Done!\n"
    clip = []
    cap = cv2.VideoCapture(args['video'])
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cur_frame = 1
    yield f"Preprocessing Video...\n"
    while True:
        flag, frame = cap.read()
        if flag:
            yield f"- {cur_frame}/{num_frame}\n"
            resorted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(resorted_frame, args['resize'])
            clip.append(resized_frame)
            cur_frame += 1
        else:
            yield f"Preprocessing Video... Done!\n"
            break
    
    yield f"Inference Action(Model, C3D)...\n"
    inference_result_list = []
    last_idx = min(args['limit_frame'], num_frame - 15)
    for start_idx in range(0, num_frame-15):
        yield f"- {start_idx}/{last_idx}\n"
        end_idx = start_idx + args['inference_frame_num']
        if end_idx < num_frame and start_idx < last_idx:
            label = inference_c3d(c3d_model, clip[start_idx:end_idx])
            inference_result_list.append(label)
        else:
            break
    yield f"Inference Action(Model, C3D)... Done!\n"
    c3d_result = max(set(inference_result_list), key=inference_result_list.count)
    yield f"Extracting Video to Frames...\n"

    # SEQ MODEL
    yield f"Initialize Seq Model...\n"
    seq_model = init_seq_model(args)
    yield f"Initialize Seq Model... Done!\n"
    yield f"Extracting Video to Frames...\n"
    _init_args = args['extract_args']['init_args']
    _call_args = args['extract_args']['call_args']
    extracted_frames = []
    inferencer = MMPoseInferencer(**_init_args)
    cap = cv2.VideoCapture(_call_args['inputs'])
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                yield f"- {frame_index}/{frame_count}\n"
                temp_call_args = _call_args
                temp_call_args['inputs'] = frame
                results = inferencer(**temp_call_args)
                frame_data = {
                    'frame_index': frame_index,
                    'objects': [],
                    'persons': []
                }

                for index, result in enumerate(results):
                    person_data = {
                        'index': index,
                        'person_index': index,
                    }
                    pred = result['predictions'][0]
                    pred.sort(key = lambda x: x['bbox'][0][0])
                    for p in pred:
                        x1, y1, x2, y2 = p['bbox'][0]
                        person_data['person_center'] = [(x1 + x2)/2, (y1 + y2)/2]
                        person_data['keypoints'] = pre_processing(p['keypoints'])
                    frame_data['persons'].append(person_data)
                extracted_frames.append(frame_data)
                frame_index += 1
            else:
                break
    cap.release()
    yield f"Extracting Video to Frames... Done!\n"
    yield f"Extracting Frames to Skeletons...\n"
    bitmap_frames = extract_skeleton(args, extracted_frames)
    yield f"Extracting Frames to Skeletons...Done!\n"
    yield f"Initialize Feature Extractor...\n"
    feature_extractor = init_feature_extractor(args)
    yield f"Initialize Feature Extractor... Done!\n"
    label_processor = tf.keras.layers.experimental.preprocessing.StringLookup(num_oov_indices=0, vocabulary=[str(key) for key in LABELMAP.keys()])
    vocabulary = label_processor.get_vocabulary()
    yield f"Inference Action(Model, SEQ)...\n"
    feature, mask = extract_feature_mask(args, feature_extractor, bitmap_frames)
    seq_result = inference_seq(seq_model, vocabulary, feature, mask)
    yield f"Inference Action(Model, SEQ)... Done!\n"

    # CATEGORIZATION
    yield f"Inference Result: C3D Model: {c3d_result}, SEQ Model: {seq_result}\n"
    yield f"Categorization Reuslts...\n"
    c3d_category = categorization(c3d_result)
    seq_category = categorization(seq_result)
    yield f"Categorization Reuslts... Done!\n"
    yield f"Categorized Result: C3D Model: {c3d_result}, SEQ Model: {seq_result}\n"

    # COMBINE
    yield f"Comine Reuslts...\n"
    result = combine(args, c3d_category, seq_category)
    yield f"Comine Reuslts... Done!\n"
    yield f"Inference Result: {result}\n"
    yield json.dumps(result)