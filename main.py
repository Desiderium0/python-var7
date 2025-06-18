import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


def load_image(path):
    """Загрузка изображения и преобразование в тензор"""
    image = Image.open(path)
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0)  # Добавляем размерность batch


def show_negative(image_tensor):
    """Отображение оригинального изображения и его негатива"""
    # Получаем негатив (1 - изображение)
    negative_tensor = 1 - image_tensor

    # Преобразуем тензоры обратно в изображения
    to_pil = transforms.ToPILImage()
    original_image = to_pil(image_tensor.squeeze(0))
    negative_image = to_pil(negative_tensor.squeeze(0))

    # Отображаем оба изображения
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Оригинал')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(negative_image)
    plt.title('Негатив')
    plt.axis('off')

    plt.show()


def select_and_show_image():
    """Открывает диалог выбора файла и показывает негатив"""
    root = tk.Tk()
    root.withdraw()  # Скрываем основное окно

    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if file_path:
        image_tensor = load_image(file_path)
        show_negative(image_tensor)


if __name__ == "__main__":
    print("Приложение для отображения негатива изображения")
    print("Пожалуйста, выберите изображение...")
    select_and_show_image()