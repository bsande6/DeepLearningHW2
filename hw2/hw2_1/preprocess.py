import cv2
import numpy as np

def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        #words = f.read().splitlines()
        words = [x.decode("utf-8").strip() for x in f.readlines()]
    return {words[i]: i for i in range(len(words))}

def video_to_frames(video):
    path = os.path.join(config.test_path, 'temporary_images')
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    video_path = os.path.join(config.test_path, 'video', video)
    count = 0
    image_list = []
    # Path to video file
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        cv2.imwrite(os.path.join(config.test_path, 'temporary_images', 'frame%d.jpg' % count), frame)
        image_list.append(os.path.join(config.test_path, 'temporary_images', 'frame%d.jpg' % count))
        count += 1

    cap.release()
    cv2.destroyAllWindows()
    return image_list

def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    return img

def extract_features(video, model):
    """
    :param video: The video whose frames are to be extracted to convert into a numpy array
    :param model: the pretrained vgg16 model
    :return: numpy array of size 4096x80
    """
    video_id = video.split(".")[0]
    print(video_id)
    print(f'Processing video {video}')

    image_list = video_to_frames(video)
    samples = np.round(np.linspace(
        0, len(image_list) - 1, 80))
    image_list = [image_list[int(sample)] for sample in samples]
    images = np.zeros((len(image_list), 224, 224, 3))
    for i in range(len(image_list)):
        img = load_image(image_list[i])
        images[i] = img
    images = np.array(images)
    fc_feats = model.predict(images, batch_size=128)
    img_feats = np.array(fc_feats)
    shutil.rmtree(os.path.join(config.test_path, 'temporary_images'))
    return img_feats