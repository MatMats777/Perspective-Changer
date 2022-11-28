import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def pair(arg):
    return [int(x) for x in arg.split(',')]

parser = argparse.ArgumentParser()
parser.add_argument("-i","--imagepath",help="The path to image")
parser.add_argument("-c","--coordinates",type=pair,nargs='+',help="Coordinates of the points which we want to change the perspective, starting from top left with clockwise orientation, separated by spaces")
args = parser.parse_args()

if args.coordinates and args.imagepath:
    img= cv2.imread(args.imagepath)

    pts1 = np.float32(
            [args.coordinates[0], 
             args.coordinates[1], 
             args.coordinates[2], 
             args.coordinates[3]] 
            )

    pts2 = np.float32(
            [[0,0], 
             [500,0], 
             [0,600], 
             [500,600]] 
            )
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    result = cv2.warpPerspective(img, matrix, (500,600))
    result = cv2.convertScaleAbs(result,alpha=1,beta=-30)

    plt.subplot(1,2,1),
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.title('Imagem Original')

    plt.subplot(1,2,2),
    plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
    plt.title('Resultado')

    plt.show()
    cv2.waitKey(0)
else:
    print("Número de argumentos inválidos")
