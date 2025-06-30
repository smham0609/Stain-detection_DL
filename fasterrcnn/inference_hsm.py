import numpy as np
import cv2
import glob
import os
import time
import yaml
import matplotlib.pyplot as plt
import pandas
import torch

from models.create_fasterrcnn_model import create_model
from utils.annotations_hsm import inference_annotations, convert_detections
from utils.general import set_infer_dir
from utils.transforms import infer_transforms, resize
from utils.logging import LogJSON

def collect_all_images(dir_test):
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images

def main():
    # 설정값 직접 정의
    input_path = 'C:/Users/CAMMSYS/Desktop/Cammsys/object_detection/fasterrcnn/dataset/val/images/.'
    output_path = None
    data_config_path = None
    model_name = None
    weights_path = 'C:/Users/CAMMSYS/Desktop/Cammsys/object_detection/fasterrcnn/outputs/training/resnet50fpn_new/best_model.pth'
    threshold = 0.3
    show = False
    mpl_show = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    imgsz = None
    no_labels = False
    square_img = False
    classes = None
    track = False
    log_json = False
    table = False

    np.random.seed(42)

    # 데이터 설정 로드
    data_configs = None
    if data_config_path is not None:
        with open(data_config_path) as file:
            data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']

    # 출력 디렉토리 설정
    if output_path is not None:
        OUT_DIR = output_path
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
    else:
        OUT_DIR = set_infer_dir()

    # 모델 로드
    if weights_path is None:
        if data_configs is None:
            with open(os.path.join('data_configs', 'test_image_config.yaml')) as file:
                data_configs = yaml.safe_load(file)
            NUM_CLASSES = data_configs['NC']
            CLASSES = data_configs['CLASSES']
        try:
            build_model = create_model[model_name]
            model, coco_model = build_model(num_classes=NUM_CLASSES, coco_model=True)
        except:
            build_model = create_model['fasterrcnn_resnet50_fpn_v2']
            model, coco_model = build_model(num_classes=NUM_CLASSES, coco_model=True)
    else:
        checkpoint = torch.load(weights_path, map_location=device)
        if data_configs is None:
            data_configs = True
            NUM_CLASSES = checkpoint['data']['NC']
            CLASSES = checkpoint['data']['CLASSES']
        try:
            print('Building from model name arguments...')
            build_model = create_model[model_name]
        except:
            build_model = create_model[checkpoint['model_name']]
        model = build_model(num_classes=NUM_CLASSES, coco_model=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device).eval()

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    
    if input_path is None:
        DIR_TEST = data_configs['image_path']
        test_images = collect_all_images(DIR_TEST)
    else:
        DIR_TEST = input_path
        test_images = collect_all_images(DIR_TEST)
    
    print(f"Test instances: {len(test_images)}")

    detection_threshold = threshold

    pred_boxes = {}
    box_id = 1

    if log_json:
        log_json = LogJSON(os.path.join(OUT_DIR, 'log.json'))

    frame_count = 0
    total_fps = 0

    for i in range(len(test_images)):
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        orig_image = cv2.imread(test_images[i])
        frame_height, frame_width, _ = orig_image.shape
        
        if imgsz is not None:
            RESIZE_TO = imgsz
        else:
            RESIZE_TO = frame_width
        
        image_resized = resize(orig_image, RESIZE_TO, square=square_img)
        image = image_resized.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = infer_transforms(image)
        image = torch.unsqueeze(image, 0)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image.to(device))
        end_time = time.time()

        fps = 1 / (end_time - start_time)
        total_fps += fps
        frame_count += 1
        
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        if log_json:
            log_json.update(orig_image, image_name, outputs[0], CLASSES)

        if len(outputs[0]['boxes']) != 0:
            draw_boxes, pred_classes, scores = convert_detections(
                outputs, detection_threshold, CLASSES, 
                {'classes': classes, 'no_labels': no_labels}
            )
            orig_image = inference_annotations(
                draw_boxes, 
                pred_classes, 
                scores, 
                CLASSES, 
                COLORS, 
                orig_image, 
                image_resized,
                track=track,
                no_labels=no_labels
            )

            if show:
                cv2.imshow('Prediction', orig_image)
                cv2.waitKey(1)
            if mpl_show:
                plt.imshow(orig_image[:, :, ::-1])
                plt.axis('off')
                plt.show()

            if table:
                for box, label in zip(draw_boxes, pred_classes):
                    xmin, ymin, xmax, ymax = box
                    width = xmax - xmin
                    height = ymax - ymin

                    pred_boxes[box_id] = {
                        "image": image_name,
                        "label": str(label),
                        "xmin": xmin,
                        "xmax": xmax,
                        "ymin": ymin,
                        "ymax": ymax,
                        "width": width,
                        "height": height,
                        "area": width * height
                    }                    
                    box_id = box_id + 1

                df = pandas.DataFrame.from_dict(pred_boxes, orient='index')
                df = df.fillna(0)
                df.to_csv(f"{OUT_DIR}/boxes.csv", index=False, sep=' ')

        cv2.imwrite(f"{OUT_DIR}/{image_name}.jpg", orig_image)
        print(f"Image {i+1} done...")
        print('-'*50)

    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()

    if log_json:
        log_json.save(os.path.join(OUT_DIR, 'log.json'))
        
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
    print('Path to output files: '+OUT_DIR)

if __name__ == '__main__':
    main()