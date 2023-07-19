#I am unsure what path the image is, so I just left a place filler. I don't know how to call the correct path from github. 

import tkinter 
from tkinter import *
from PIL import ImageTk, Image 
import gc

#initiating the window
root = Tk()
root.title = ("Chat Bot")
root.geometry = ("800x2000")
root.resizable(width = True, height = True)

main_menu = Menu(root)
file_menu = Menu(root)

#ngl i don't know why they made these 
file_menu.add_command(label="New..")
file_menu.add_command(label="Save As..")
file_menu.add_command(label="Exit")
main_menu.add_cascade(label="File", menu=file_menu)
main_menu.add_command(label="Edit")
main_menu.add_command(label="Quit")
root.config(menu=main_menu)

# placing all the blocks, needs future change for automatic resizing

# the fpl logo and basically left chunk
pictureWindow = Text(root, bd = 1, bg = "gray", width = "50", height = "8", font = ("Arial", 50), foreground = "#00ffff")
pictureWindow.place(x = 6, y = 6, height = 760, width = 500)

pictureWindow.insert(INSERT, "FPL CHATBOT")

#img code
img = ImageTk.PhotoImage(Image.open('/FPL_Datasets/gui/fpl.jpg'))

img_new = img._PhotoImage__photo.subsample(3,3)

panel = Label(image = img_new)

panel.place(x = 50, y = 200)

#displays all chats
chatWindow = Text(root, bd = 1, bg = "black", width = "50", height = "8", font = ("Arial", 18), foreground = "#00ffff")
chatWindow.place(x = 512, y = 6, height = 600, width = 1000)

#placing the scrollbar
scrollbar = Scrollbar(root, command=chatWindow.yview, cursor="mouse")
scrollbar.place(x = 1512, y = 6, height = 600)

#the window where the messages are inputted
messageWindow = Text(root, bd=0, bg="black",width="30", height="4", font=("Arial", 18), foreground="#ffffff")
messageWindow.place(x = 512, y = 612, height = 154, width = 750)

# the button to send
sendButton= Button(root, text="Send",  width="12", height=5,
                    bd=0, bg="#0080ff", activebackground="#00bfff",foreground='#ffffff',font=("Arial", 16))
sendButton.place(x=1268, y=612, height=154, width = 250)

root.mainloop()
