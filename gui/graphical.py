import tkinter as tk
# from tkinter import tkk
from PIL import ImageTk, Image 
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import gc
import datetime
import os


# Model: Simple Echo Chatbot
class SimpleChatbot():
    def get_response(self, message):
        return f"Bot: {message}"

class window(tk.Tk):
    def __init__(self):
        super().__init__()
        #initiating the window
        self.title = ("Chat Bot")
        self.geometry('1160x582')
        self.resizable(width = False, height = False)
        
        self.chatbot = SimpleChatbot()
        
        #display pull-down and pop-up menu widget
        self.main_menu = tk.Menu(self)
        self.main_menu.add_command(label= 'Clear', command = self.clearScreen)
        #Add load history command
        
        # the fpl logo and basically left chunk
        self.pictureWindow = tk.Text(self, bd = 1, bg = "gray", width = "50", height = "8", font = ("Times New Roman", 30), foreground = "#00ffff", padx = 12, pady = 6)
        self.pictureWindow.place(x = 6, y = 6, height = 570, width = 375)
        self.pictureWindow.insert(tk.INSERT, "FPL \nCHATBOT")
        
        # Load the image using Pillow (PIL)
        # self.img = ImageTk.PhotoImage(Image.open('/Users/raulmendy/Desktop/FPL/Datasets/FPL_Datasets/assets/fpl.jpg'))
        # self.img_new = self.img.resize((300, 300))
        img = Image.open('/Users/raulmendy/Desktop/FPL/Datasets/FPL_Datasets/assets/fpl.jpg')
        img_new = img.resize((200, 230))  # Resize the image if needed
        self.photo = ImageTk.PhotoImage(img_new)
        # Create a Label to display the image
        self.frame = tk.Frame(self, width=200, height= 230)
        self.frame.place(x=70, y=230)
        
        self.label = tk.Label(self.frame,image=self.photo)
        self.label.pack()
        
        #displays all chats
        self.chatWindow = tk.Text(self, bd = 1, bg = "black", width = "50", height = "8", font = ("Times New Roman", 14), foreground = "#00ffff")
        self.chatWindow.place(x = 387, y = 6, height = 450, width = 750)
        
        #placing the scrollbar
        self.scrollbar = tk.Scrollbar(self, command=self.chatWindow.yview, cursor="mouse")
        self.scrollbar.place(x = 1137, y = 6, height = 450)
        
        #the window where the messages are inputted
        self.messageWindow = tk.Text(self, bd=0, bg="black", width="30", height="4", font=("Times New Roman", 18), foreground="#ffffff")
        self.messageWindow.place(x = 387, y = 462, height = 115, width = 562)
        
        # open button
        self.open_button = tk.Button(self,text='Open a File',command=self.select_file)
        self.open_button.place(x = 125, y = 500)
        
        # the button to send
        self.sendButton= tk.Button(self, text="Send",  width="12", height=5,bd=0, bg="#0080ff", activebackground="#00bfff",foreground='black',font=("Arial", 16), command = self.send_message)
        self.sendButton.place(x= 955, y=462, height= 115, width = 188)
        #chat history file
        self.history_file = open("chat_history.txt", "a")

                
        
    def send_message(self):
        user_message = self.messageWindow.get("1.0", "end").strip()
        if user_message:
            response = self.chatbot.get_response(user_message)
            self.display_message(f"User: {user_message}")
            self.display_message(response, bot=True)

            # Save chat history with timestamp header to the file
            time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.history_file.write(f"\n----- Session: {time_now} -----\n")
            self.history_file.write(f"User: {user_message}\n")
            self.history_file.write(f"Bot: {response}\n")
            self.history_file.flush()  # Flush to ensure data is written immediately

            self.messageWindow.delete("1.0", "end")
        else:
            pass
            # messagebox.showinfo("Error", "Please enter a message.")

    def display_message(self, message, bot=False):
        if bot:
            self.chatWindow.tag_configure("bot", foreground="green")
            message = "\n" + message
            self.chatWindow.insert(tk.END, message, "bot")
        else:
            self.chatWindow.tag_configure("user", foreground="blue")
            self.chatWindow.insert(tk.END, "\n" + message, "user")

    def __del__(self):
        # Close the chat history file when the GUI is closed
        self.history_file.close()
            
    def clearScreen(self):
        self.chatWindow.delete('1.0', 'end')
        
    #file selection
    def select_file(self):
        self.filetypes = (('text files', '*.txt'),('All files', '*.*'))
        self.filepath = fd.askopenfilename(title='Open a file',initialdir='/',filetypes=self.filetypes)
        showinfo(title='Selected File',message=self.filepath)
        try: 
            self.filename = os.path.basename(self.filepath)
            self.text_widget = self.nametowidget(self.textcon)
            with open(self.filepath, "r") as file:
                self.content=file.read()
                self.textcon.delete('1.0', 'end-1c')
                self.text_contents[str(self.textcon)] = hash(self.content)
                self.text_widget.insert(tk.END,self.content)
            print("Operation successfull")
            return self.filename
        except(FileNotFoundError):
            print("Operation not successfull")
            return None
            
        

if __name__ == "__main__":
    testObj = window()
    testObj.mainloop()
