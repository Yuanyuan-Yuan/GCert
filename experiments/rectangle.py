############################################################
# This script shows how the minimal enclosing rectangle   #
# is obtained in our first evaluation.                     #
############################################################

import cv2
from PIL import Image
import torchvision.transforms as transforms

def to_image(tensor):
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).squeeze(2).cpu().numpy()
    return Image.fromarray(ndarr).convert('RGB')

def to_ndarr(tensor):
    # ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).squeeze(2).cpu().numpy()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return ndarr

generator = None
z = None
# here, you can use any generator and z

G = generator(z)
ndarr = to_ndarr(G).astype(np.uint8)

ndarr = cv2.cvtColor(ndarr, cv2.COLOR_GRAY2BGR)
ret, thresh = cv2.threshold(ndarr, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, 1, 2)[-2:]

rect = cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(ndarr, [box], 0, (0, 255, 0), 2)

cv2.imwrite('image_with_rectangle.jpg', ndarr)