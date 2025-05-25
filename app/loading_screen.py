# loading_screen.py
import tkinter as tk
import multiprocessing

class IndeterminateProgressBar:
    def __init__(self, master, width=300, height=10, bg_color="#042a2b", fg_color="#4CAF50"):
        self.master = master
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.fg_color = fg_color

        self.canvas = tk.Canvas(master, width=self.width, height=self.height, bg=self.bg_color, bd=0, highlightthickness=0)
        self.canvas.pack(pady=20)

        self.bg_rect = self.canvas.create_rectangle(0, 0, self.width, self.height, fill=self.bg_color, outline="")
        self.fg_rect = self.canvas.create_rectangle(0, 0, self.width / 4, self.height, fill=self.fg_color, outline="")

        self.animation_speed = 20
        self.x_position = 0

    def animate(self):
        self.x_position += 5
        if self.x_position > self.width:
            self.x_position = -self.width / 4
        self.canvas.coords(self.fg_rect, self.x_position, 0, self.x_position + self.width / 4, self.height)
        self.master.after(self.animation_speed, self.animate)

def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")

def run_loading_screen():
    root = tk.Tk()
    root.overrideredirect(True)
    root.configure(bg="#1e1e1e")
    center_window(root, 400, 170)

    label = tk.Label(root, text="Opening Camvas", font=("Poppins", 14), bg="#1e1e1e", fg="white")
    label.pack(pady=(40, 0))

    bar = IndeterminateProgressBar(root, bg_color="#3a3a3a", fg_color="#00FFAA")
    root.after(100, bar.animate)

    root.mainloop()

def start():
    multiprocessing.freeze_support()
    global loader_process
    loader_process = multiprocessing.Process(target=run_loading_screen)
    loader_process.start()

def stop():
    loader_process.terminate()
    loader_process.join()