import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("800x600")

        self.image_tensor = None
        self.original_image = None

        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Image selection buttons
        select_frame = ttk.LabelFrame(main_frame, text="Выбор изображения", padding="10")
        select_frame.pack(fill=tk.X, pady=5)

        ttk.Button(select_frame, text="Загрузить из файла", command=self.load_from_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_frame, text="Сделать снимок", command=self.capture_from_webcam).pack(side=tk.LEFT, padx=5)

        # Image display
        self.display_frame = ttk.LabelFrame(main_frame, text="Изображение", padding="10")
        self.display_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Processing options
        process_frame = ttk.LabelFrame(main_frame, text="Опции обработки", padding="10")
        process_frame.pack(fill=tk.X, pady=5)

        # Color channel selection
        channel_frame = ttk.Frame(process_frame)
        channel_frame.pack(fill=tk.X, pady=5)

        ttk.Label(channel_frame, text="Цветовой канал:").pack(side=tk.LEFT)
        self.channel_var = tk.StringVar(value="None")
        ttk.Radiobutton(channel_frame, text="Красный", variable=self.channel_var, value="R").pack(side=tk.LEFT)
        ttk.Radiobutton(channel_frame, text="Зеленый", variable=self.channel_var, value="G").pack(side=tk.LEFT)
        ttk.Radiobutton(channel_frame, text="Синий", variable=self.channel_var, value="B").pack(side=tk.LEFT)
        ttk.Button(channel_frame, text="Применить", command=self.show_color_channel).pack(side=tk.LEFT, padx=10)

        # Rotation
        rotate_frame = ttk.Frame(process_frame)
        rotate_frame.pack(fill=tk.X, pady=5)

        ttk.Label(rotate_frame, text="Угол поворота:").pack(side=tk.LEFT)
        self.angle_entry = ttk.Entry(rotate_frame, width=10)
        self.angle_entry.pack(side=tk.LEFT)
        ttk.Button(rotate_frame, text="Повернуть", command=self.rotate_image).pack(side=tk.LEFT, padx=10)

        # Negative image
        ttk.Button(process_frame, text="Негатив", command=self.show_negative).pack(pady=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X)

        # Circle drawing controls
        circle_frame = ttk.Frame(process_frame)
        circle_frame.pack(fill=tk.X, pady=5)

        ttk.Label(circle_frame, text="Круг (x,y,r):").pack(side=tk.LEFT)
        self.circle_x_entry = ttk.Entry(circle_frame, width=5)
        self.circle_x_entry.pack(side=tk.LEFT)
        self.circle_y_entry = ttk.Entry(circle_frame, width=5)
        self.circle_y_entry.pack(side=tk.LEFT)
        self.circle_r_entry = ttk.Entry(circle_frame, width=5)
        self.circle_r_entry.pack(side=tk.LEFT)
        ttk.Button(circle_frame, text="Нарисовать", command=self.draw_circle).pack(side=tk.LEFT, padx=10)

        self.update_status("Готов к работе")

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def load_from_file(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Выберите изображение",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
            )

            if file_path:
                self.update_status(f"Загрузка изображения: {file_path}")
                self.original_image = Image.open(file_path)
                self.image_tensor = transforms.ToTensor()(self.original_image).unsqueeze(0)
                self.show_image(self.original_image)
                self.update_status(f"Изображение загружено: {file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{str(e)}")
            self.update_status("Ошибка загрузки изображения")

    def capture_from_webcam(self):
        try:
            self.update_status("Попытка подключения к веб-камере...")
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                raise RuntimeError("Не удалось подключиться к веб-камере")

            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise RuntimeError("Не удалось получить изображение с камеры")

            self.update_status("Обработка изображения с камеры...")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.original_image = Image.fromarray(frame)
            self.image_tensor = transforms.ToTensor()(self.original_image).unsqueeze(0)
            self.show_image(self.original_image)
            self.update_status("Изображение с камеры получено")
        except Exception as e:
            messagebox.showerror("Ошибка",
                                 f"Проблема с веб-камерой:\n{str(e)}\n\n"
                                 "Возможные решения:\n"
                                 "1. Проверьте подключение камеры\n"
                                 "2. Убедитесь, что камера не используется другим приложением\n"
                                 "3. Проверьте права доступа для приложения\n"
                                 "4. Попробуйте перезапустить приложение")
            self.update_status("Ошибка захвата с камеры")

    def show_image(self, image):
        # Clear previous image
        for widget in self.display_frame.winfo_children():
            widget.destroy()

        # Display new image
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()

        # Embed matplotlib figure in tkinter
        canvas = FigureCanvasTkAgg(plt.gcf(), master=self.display_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_color_channel(self):
        if self.image_tensor is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        channel = self.channel_var.get()
        if channel == "None":
            messagebox.showwarning("Предупреждение", "Выберите цветовой канал")
            return

        try:
            self.update_status(f"Применение {channel} канала...")

            # Create a blank tensor with same shape
            channel_image = torch.zeros_like(self.image_tensor)

            # Set selected channel
            if channel == "R":
                channel_image[:, 0, :, :] = self.image_tensor[:, 0, :, :]
            elif channel == "G":
                channel_image[:, 1, :, :] = self.image_tensor[:, 1, :, :]
            elif channel == "B":
                channel_image[:, 2, :, :] = self.image_tensor[:, 2, :, :]

            # Convert to PIL image
            to_pil = transforms.ToPILImage()
            processed_image = to_pil(channel_image.squeeze(0))

            self.show_image(processed_image)
            self.update_status(f"Отображен {channel} канал изображения")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обработать канал:\n{str(e)}")
            self.update_status("Ошибка обработки канала")

    def rotate_image(self):
        if self.image_tensor is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        try:
            angle = float(self.angle_entry.get())
            self.update_status(f"Поворот на {angle} градусов...")

            # Rotate using PIL
            rotated_image = self.original_image.rotate(angle, expand=True)

            self.show_image(rotated_image)
            self.update_status(f"Изображение повернуто на {angle} градусов")
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректное число для угла поворота")
            self.update_status("Ошибка: некорректный угол поворота")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось повернуть изображение:\n{str(e)}")
            self.update_status("Ошибка поворота изображения")

    def show_negative(self):
        if self.image_tensor is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        try:
            self.update_status("Создание негатива...")

            # Create negative (1 - image)
            negative_tensor = 1 - self.image_tensor

            # Convert to PIL image
            to_pil = transforms.ToPILImage()
            negative_image = to_pil(negative_tensor.squeeze(0))

            self.show_image(negative_image)
            self.update_status("Негатив изображения создан")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать негатив:\n{str(e)}")
            self.update_status("Ошибка создания негатива")

    def draw_circle(self):
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        try:
            # Get circle parameters from user input
            x = int(self.circle_x_entry.get())
            y = int(self.circle_y_entry.get())
            r = int(self.circle_r_entry.get())

            self.update_status(f"Рисование круга в ({x},{y}) радиусом {r}...")

            # Convert image to RGB if it's not
            image = self.original_image.convert("RGB")

            # Create drawable image
            drawable_image = image.copy()
            draw = ImageDraw.Draw(drawable_image)

            # Draw red circle (outline only)
            draw.ellipse([(x - r, y - r), (x + r, y + r)], outline="red", width=3)

            # Update displayed image
            self.show_image(drawable_image)
            self.update_status(f"Круг нарисован в ({x},{y}) радиусом {r}")

        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числа для координат и радиуса")
            self.update_status("Ошибка: некорректные параметры круга")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось нарисовать круг:\n{str(e)}")
            self.update_status("Ошибка рисования круга")

if __name__ == "__main__":
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        root = tk.Tk()
        app = ImageProcessorApp(root)
        root.mainloop()
    except ImportError as e:
        print(f"Ошибка импорта: {str(e)}")
        print("Убедитесь, что установлены все зависимости:")
        print("pip install torch torchvision pillow matplotlib opencv-python")