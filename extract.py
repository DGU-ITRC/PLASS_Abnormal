import cv2
import json
from mmpose.apis.inferencers import MMPoseInferencer

filter_args = dict(bbox_thr=0.3, nms_thr=0.3, pose_based_nms=False)
CLEANER = ' ' * 30 + '\n'
POSE2D_SPECIFIC_ARGS = dict(
    yoloxpose=dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True),
    rtmo=dict(bbox_thr=0.65, nms_thr=0.65, pose_based_nms=True),
)

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
    print('\033[1;32mInfrence Action...\033[0m', end='\r', flush=True)
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

if __name__ == '__main__':
    video_path = "./test.mp4"
    file_data = extract_video(video_path)
    with open("./file_data.json", 'w') as f:
        json.dump(file_data, f)
    

