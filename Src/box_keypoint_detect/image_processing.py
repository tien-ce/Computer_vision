import cv2
import numpy as np

# f(y,x) = [B,G,R]
def apply_point_processing (img, h, w):
    for y in range (h):
        for x in range (w):
            if y % 20 < 5: 
                # Create the red horizonal striped (5 pixels width) every 20 pixels 
                img[y,x] = [0,0,255]
            elif x % 40 < 5:
                # Create a blue vertical stripped every 40 pixels
                img[y,x] = [255,0,0]

def main():
    image = cv2.imread("D:\Van_Tien\Project\Camera\Input\human_detections\Screenshot 2026-06-19 130928.png")
    height,width,channel = image.shape
    print (type(image))
    cv2.imshow("Original",image)
    # Create a singel-chanel (nd array) blank black mask 
    mask = np.zeros((height,width),dtype = np.uint8)
    cv2.imshow("Mask",mask)
    # Create an array holding points of polygon we want to crop [[X,Y]] <---> [width,height]
    polygon_poitns = np.array([
                                [width // 4, height // 4], # Top-left
                                [3 * width // 4, height // 4], # Top-right
                                [3 * width // 4, 3 * height // 4], # Bottom-Right
                                [width // 4, 3 * height // 4]    # Bottom-Left
                            ],dtype=np.int32)
    cv2.fillConvexPoly(mask,polygon_poitns,255)
    cv2.imshow("Mask after fill convexpoy",mask)
    mask_3d = mask[:, :, np.newaxis]
    #masked_image = cv2.bitwise_and(image,image,mask=mask)
    masked_image = image & mask_3d
    cv2.imshow("Image after masked", masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows
if __name__ == "__main__":
    main()
