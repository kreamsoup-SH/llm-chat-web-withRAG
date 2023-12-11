'''
Realtime KoLLaVA (RKoL)
Adaptation of the KoLLaVA model for real-time use
'''

import cv2
from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained("tabtoyou/KoLLaVA-v1.5-Synatra-7b")
model = AutoModelForCausalLM.from_pretrained("tabtoyou/KoLLaVA-v1.5-Synatra-7b")


def camera(w:int = 640, h:int = 480):
    cap = cv2.VideoCapture(0)
    print("original frame size: ", cap.get(3), cap.get(4))
    cap.set(3, w)
    cap.set(4, h)
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
