# -*- coding:utf-8 -*-
# @TIME     :2018/12/28 15:36
# @File     :gui_test.py
import tkinter as tk
from tkinter import filedialog
import threading


class Gui:
    def __init__(self):
        self.filepath = '/home/fan/Pose Estimation/Demo/pic.jpg'
        self.id = 0         # 0 means image, 1 means video
        gui = tk.Tk(className='Pose Estimation')
        gui.geometry("640x320")

        # set the display menu
        menubar = tk.Menu(gui)

        filemenu = tk.Menu(menubar, tearoff=False)
        filemenu.add_command(label="Image", command=self.get_image_file)
        filemenu.add_command(label="Video", command=self.get_video_file)
        menubar.add_cascade(label="Load", menu=filemenu)


        runmenu = tk.Menu(menubar, tearoff=False)
        runmenu.add_command(label="run", command=gui.quit)
        menubar.add_cascade(label="Run", menu=runmenu)

        gui.config(menu=menubar)

        # set the display text
        string = '\nThanks for using~\n\nClick Load to choose the file\n' \
                 '\nClick Run to begin the precess\n\nDesigned by YuHan\n' \
                 '\nHave a good day!'
        label = tk.Label(gui, text=string, font=20)
        label.pack()
        #tk.mainloop()

    def get_image_file(self):
        self.filepath = filedialog.askopenfilename()
        self.id = 0

    def get_video_file(self):
        self.filepath = filedialog.askopenfilename()
        self.id = 1

    def begin(self):
        tk.mainloop()

    def thread(self, func, *args):
        t = threading.Thread(target=func, args=args)
        t.setDaemon(True)
        t.start()


