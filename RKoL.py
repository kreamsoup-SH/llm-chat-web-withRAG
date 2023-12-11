'''
Realtime KoLLaVA (RKoL)
Adaptation of the KoLLaVA model for real-time use
'''

'''
Imports packages
'''
import cv2
import time
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

'''
# KoLLaVA / LLaVA Model Loading
'''
# processor = AutoProcessor.from_pretrained("tabtoyou/KoLLaVA-v1.5-Synatra-7b")
# model = AutoModelForCausalLM.from_pretrained("tabtoyou/KoLLaVA-v1.5-Synatra-7b")
model_id = "llava-hf/llava-1.5-7b-hf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

'''
# KoLLaVA / LLaVA Inference
'''
def inference(pipe, image, prompt):
    max_new_tokens = 200
    prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    return outputs[0]["generated_text"]


'''
# Video Capture from camera
'''
class Camera():
    def __init__(self, w:int = 640, h:int = 480, frame_rate:int = 5):
        self.cap = cv2.VideoCapture(0)
        print("original frame size: ", self.cap.get(3), self.cap.get(4))
        self.cap.set(3, w)
        self.cap.set(4, h)
        self.frame_rate = frame_rate
        self.prev = 0
    
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
    
    def capture(self):
        time_elapsed = time.time() - self.prev
        res, image = self.cap.read()
        if res and time_elapsed > 1./self.frame_rate:
            self.prev = time.time()
            return image
        time.sleep(max(1./self.frame_rate - time_elapsed, 0))


'''
Main function
'''
def main():
    pass

if __name__ == "__main__":
    main()