import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from predict_img import Predicter

class AppWindow:

    def __init__(self):
        self.root = tk.Tk()
        self.button = tk.Button(self.root, text='Wczytaj obraz', command=self.load_image)
        self.button.pack()

        self.label = tk.Label(self.root)
        self.label.pack()

        self.status = tk.Label(self.root, text='', bd=1, relief='sunken', anchor='w')
        self.status.pack(side='bottom', fill='x')

    def load_image(self):
        predicter = Predicter()
        # Wczytanie i przetworzenie obrazu
        image_path = filedialog.askopenfilename()
        image = Image.open(image_path)
        if image.height > 400:
            # Oblicz nową szerokość zachowując proporcje
            new_width = int((image.width / image.height) * 400)
            image = image.resize((new_width, 400))
        image_disp = ImageTk.PhotoImage(image)
        self.label.config(image=image_disp)
        self.label.image = image_disp

        self.status.config(text='Proszę czekać, trwa analizowanie...')
        self.root.update()

        result = predicter.predict(image_path)
        self.status.config(text=result)

    def run(self):
        self.root.mainloop()

