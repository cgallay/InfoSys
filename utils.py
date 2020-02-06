import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} (Avg: {avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
def crop_face(img, detector):
    """
        Detect face from image and crop it.
    """
    faces = detector.detect_faces(img)
    assert len(faces) == 1, "No or more than one faces detected"
    f_cord = faces[0]['box']
    face_croped = img[f_cord[1]: f_cord[1] + f_cord[3], f_cord[0]: f_cord[0] + f_cord[2], :]
    # grayscale
    img_resize = cv2.resize(face_croped, (48, 48))
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    return img_gray

def load_img(path):
    """
        Load an image from disk
    """
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def get_emotion_from_image(path, fer_net, face_detector):
    """
        Show image and the probability asociated with each category
    """
    img = load_img(path)
    plt.imshow(img)
    plt.show()
    face = crop_face(img, face_detector)
    #print(face)
    predict_emotion(face, fer_net)

def show_proba(probas):
    """
        Print probability in a sorted way
    """
    print({code2emotion[i]: p for i, p in reversed(sorted(enumerate(probas), key=lambda x: x[1]))})

def predict_emotion(img_gray, fer_net):
    """
        Print probability associated with the image given in first argument
    """
    assert img_gray.shape == (48, 48), "Image should be of shape (48, 48)"
    tens = torch.from_numpy(img_gray)
    tens = tens.unsqueeze(0).unsqueeze(0).float() / 255.0
    tens = tens.to(device)
    with torch.no_grad():
        probas = nn.Softmax()(fer_net(tens)).cpu().numpy()[0]
    show_proba(probas)

    
code2emotion = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

