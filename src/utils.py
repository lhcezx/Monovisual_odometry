import os
from config import parse_configs
import matplotlib.pyplot as plt
import numpy as np
import cv2
configs = parse_configs()


def rename_image(path):
    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file):
            os.rename(os.path.join(path, file),
                      os.path.join(path, file.split("_")[1]))


def matrix2val(cam_matrix):
    fx = cam_matrix[0, 0]
    fy = cam_matrix[1, 1]
    cx = cam_matrix[0, 2]
    cy = cam_matrix[1, 2]
    return fx, fy, cx, cy


def visualize_flux(rgb, kp1, kp2):
    
    if isinstance(rgb, np.ndarray):
        color = np.random.randint(0, 255, (100000, 3))
        mask = np.zeros_like(rgb)
        for i, (new, old) in enumerate(zip(kp2, kp1)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)),
                            (int(c), int(d)), color[i].tolist(), 1)
            rgb = cv2.circle(rgb, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(rgb, mask)
        cv2.imshow('frame', img)
        cv2.waitKey(0) 


def histogram():
    label_list = ['Fast', 'Orb', 'Sift', "Shi-Tomasi"]
    num_list1 = [0.0066822000000001935, 0.3015093999999996,
                 1.4251822999999995, 0.20651249999999965]
    x = range(len(num_list1))

    rects1 = plt.bar(x, height=num_list1, width=0.2,
                     alpha=0.4, color='red', align="center")
    plt.ylabel("Temps en second s")
    plt.xticks([index for index in x], label_list)
    plt.xlabel("Algorithmes")
    plt.title("Temps de détection pour une image")
    plt.show()

def resize(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (300,300))
    cv2.imwrite(path, img)

if __name__ == "__main__":
    resize("src/Traj_ORB_nuit_amelioration.png")
