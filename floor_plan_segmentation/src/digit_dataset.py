import numpy as np
from torch.utils.data import Dataset
import cv2


class CustomDigitDataset(Dataset):
    def __init__(self, num_images, letters=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'], size_low=1.2, size_high=1.5):
        self.images = []
        self.labels = []
        self.letters = letters

        for i in range(num_images):
            num = np.random.randint(0, len(letters))

            s = np.random.default_rng().uniform(size_low, size_high)
            x = np.random.randint(0, 18 // s)
            y = np.random.randint(0, 18 // s)

            x1 = np.random.randint(0, 24 // s)
            y1 = np.random.randint(0, 24 // s)
            s1 = np.random.randint(0, 5)
            s2 = np.random.randint(0, 5)

            thick = np.random.randint(0, 4)
            font = [cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_COMPLEX][np.random.randint(0,2)]

            digit = np.zeros((28, 28), dtype=np.float32)
            digit = cv2.putText(digit, letters[num], (x, 28 - y),  font, s, [255.0, 255.0, 255.0, 1.0], thick)

            digit[y1:y1+s1, x1:x1+s2] = 0

            digit = cv2.GaussianBlur(digit, (3, 3), np.random.random(1)[0])
            digit = np.reshape(digit, (1, 28, 28))
            digit = digit / 255.

            self.labels.append(num)
            self.images.append(digit)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
