import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import openpyxl  # For reading .xlsx files
import xlrd  # For reading .xls files

import algo_final as algo
import cleaning_moh_final as cleaning
import visualizer
import instructor_visualizer

class CourseSchedulerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Course Scheduler")
        self.geometry("600x400")
        self.configure(bg="#D0D3D4")
        self.center_window()

        self.create_widgets()

    def create_widgets(self):
        # Create frames for each section
        self.main_menu_frame = tk.Frame(self, bg="#D0D3D4")
        self.cleaning_frame = tk.Frame(self, bg="#D0D3D4")
        self.schedule_frame = tk.Frame(self, bg="#D0D3D4")
        self.classroom_visualizer_frame = tk.Frame(self, bg="#D0D3D4")
        self.instructor_visualizer_frame = tk.Frame(self, bg="#D0D3D4")

        # Initialize with main menu frame visible
        self.main_menu_frame.pack(fill='both', expand=1)

        self.create_main_menu_widgets()
        self.create_cleaning_widgets()
        self.create_schedule_widgets()
        self.create_classroom_visualizer_widgets()
        self.create_instructor_visualizer_widgets()

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def center_popup(self, window):
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def create_main_menu_widgets(self):
        title_label = tk.Label(self.main_menu_frame, text="Course Scheduler", font=("Arial", 18, "bold"), bg="#D0D3D4")
        title_label.pack(pady=20)

        button_width = 25  # Set a standard width for all buttons
        button_height = 2  # Set a standard height for all buttons

        btn_clean_excel = tk.Button(self.main_menu_frame, text="Clean Excel File", command=self.show_cleaning_frame, bg="#FFFFFF", width=button_width, height=button_height)
        btn_clean_excel.pack(pady=5)

        btn_generate_schedule = tk.Button(self.main_menu_frame, text="Generate Course Schedule", command=self.show_schedule_frame, bg="#FFFFFF", width=button_width, height=button_height)
        btn_generate_schedule.pack(pady=5)

        btn_classroom_visualizer = tk.Button(self.main_menu_frame, text="Display Classroom Visualizer", command=self.show_classroom_visualizer_frame, bg="#FFFFFF", width=button_width, height=button_height)
        btn_classroom_visualizer.pack(pady=5)

        btn_instructor_visualizer = tk.Button(self.main_menu_frame, text="Display Instructor Visualizer", command=self.show_instructor_visualizer_frame, bg="#FFFFFF", width=button_width, height=button_height)
        btn_instructor_visualizer.pack(pady=5)

        btn_exit = tk.Button(self.main_menu_frame, text="Exit", command=self.quit, bg="#FFFFFF", width=button_width, height=button_height)
        btn_exit.pack(pady=20)

    def create_cleaning_widgets(self):
        file_label = tk.Label(self.cleaning_frame, text="Enter the path to your Excel file:", bg="#D0D3D4")
        file_label.pack(pady=5)

        file_btn = tk.Button(self.cleaning_frame, text="Select Excel File", command=self.clean_excel_file, bg="#FFFFFF")
        file_btn.pack(pady=5)

        # Using pack for consistency within the cleaning_frame
        self.back_to_menu_btn(self.cleaning_frame).pack(pady=5)

    def create_schedule_widgets(self):
        # Layout using grid for precise control
        self.schedule_frame.grid_columnconfigure(0, weight=1)

        gen_label = tk.Label(self.schedule_frame, text="Enter the generation number:", bg="#D0D3D4")
        gen_label.grid(row=0, column=0, pady=5)
        self.gen_entry = tk.Entry(self.schedule_frame, width=30)
        self.gen_entry.grid(row=1, column=0, pady=5)

        spring_btn = tk.Button(self.schedule_frame, text="Select Spring CSV File", command=self.select_spring_file, bg="#FFFFFF")
        spring_btn.grid(row=2, column=0, pady=5)

        self.spring_file_label = tk.Label(self.schedule_frame, text="No file selected", bg="#D0D3D4")
        self.spring_file_label.grid(row=3, column=0, pady=5)

        fall_btn = tk.Button(self.schedule_frame, text="Select Fall CSV File", command=self.select_fall_file, bg="#FFFFFF")
        fall_btn.grid(row=4, column=0, pady=5)

        self.fall_file_label = tk.Label(self.schedule_frame, text="No file selected", bg="#D0D3D4")
        self.fall_file_label.grid(row=5, column=0, pady=5)

        self.progress_label = tk.Label(self.schedule_frame, text="", bg="#D0D3D4")
        self.progress_label.grid(row=6, column=0, pady=10)

        self.progress_bar = ttk.Progressbar(self.schedule_frame, orient="horizontal", length=250, mode="determinate")
        self.progress_bar.grid(row=7, column=0, pady=(0, 20))  # Separate the progress bar from the OK button

        self.ok_btn = tk.Button(self.schedule_frame, text="OK", command=self.start_generation, bg="#FFFFFF")
        self.ok_btn.grid(row=8, column=0, pady=10)  # Place the OK button under the progress bar

        self.back_btn = self.back_to_menu_btn(self.schedule_frame)
        self.back_btn.grid(row=9, column=0, pady=10)

    def create_classroom_visualizer_widgets(self):
        semester_label = tk.Label(self.classroom_visualizer_frame, text="Enter the semester (spring/fall):", bg="#D0D3D4")
        semester_label.pack(pady=5)
        self.semester_entry = tk.Entry(self.classroom_visualizer_frame, width=30)
        self.semester_entry.pack(pady=5)

        visualize_btn = tk.Button(self.classroom_visualizer_frame, text="Visualize", command=self.display_classroom_visualizer, bg="#FFFFFF")
        visualize_btn.pack(pady=20)

        self.back_to_menu_btn(self.classroom_visualizer_frame).pack(pady=5)

    def create_instructor_visualizer_widgets(self):
        instructor_label = tk.Label(self.instructor_visualizer_frame, text="Enter the instructor name:", bg="#D0D3D4")
        instructor_label.pack(pady=5)
        self.instructor_entry = tk.Entry(self.instructor_visualizer_frame, width=30)
        self.instructor_entry.pack(pady=5)

        semester_label = tk.Label(self.instructor_visualizer_frame, text="Enter the semester (spring/fall):", bg="#D0D3D4")
        semester_label.pack(pady=5)
        self.instructor_semester_entry = tk.Entry(self.instructor_visualizer_frame, width=30)
        self.instructor_semester_entry.pack(pady=5)

        visualize_btn = tk.Button(self.instructor_visualizer_frame, text="Visualize", command=self.display_instructor_visualizer, bg="#FFFFFF")
        visualize_btn.pack(pady=20)

        self.back_to_menu_btn(self.instructor_visualizer_frame).pack(pady=5)

    def back_to_menu_btn(self, frame):
        return tk.Button(frame, text="Back to Main Menu", command=self.show_main_menu_frame, bg="#FFFFFF")

    def show_frame(self, frame):
        self.main_menu_frame.pack_forget()
        self.cleaning_frame.pack_forget()
        self.schedule_frame.pack_forget()
        self.classroom_visualizer_frame.pack_forget()
        self.instructor_visualizer_frame.pack_forget()

        frame.pack(fill='both', expand=1)
        self.center_window()

    def show_main_menu_frame(self):
        self.show_frame(self.main_menu_frame)

    def show_cleaning_frame(self):
        self.show_frame(self.cleaning_frame)

    def show_schedule_frame(self):
        self.show_frame(self.schedule_frame)

    def show_classroom_visualizer_frame(self):
        self.show_frame(self.classroom_visualizer_frame)

    def show_instructor_visualizer_frame(self):
        self.show_frame(self.instructor_visualizer_frame)

    def clean_excel_file(self):
        file_path = filedialog.askopenfilename(title="Select Excel File", filetypes=[("Excel files", "*.xlsx;*.xls")])
        if file_path:
            sheet_names = self.get_sheet_names(file_path)
            if sheet_names:
                sheet_name = self.select_sheet_name(sheet_names)
                if sheet_name:
                    try:
                        cleaning.process_and_save_excel_file(file_path, sheet_name)
                        messagebox.showinfo("Success", f"Excel file cleaned successfully. Sheet: {sheet_name}")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to clean the Excel file.\nError: {str(e)}")

    def get_sheet_names(self, file_path):
        if file_path.endswith('.xlsx'):
            workbook = openpyxl.load_workbook(file_path, read_only=True)
            return workbook.sheetnames
        elif file_path.endswith('.xls'):
            workbook = xlrd.open_workbook(file_path)
            return workbook.sheet_names()
        else:
            messagebox.showerror("Error", "Unsupported file format.")
            return None

    def select_sheet_name(self, sheet_names):
        sheet_name = tk.StringVar()

        def on_select():
            selected_sheet = sheet_listbox.get(tk.ACTIVE)
            sheet_name.set(selected_sheet)
            select_window.destroy()

        select_window = tk.Toplevel(self)
        select_window.title("Select Sheet Name")
        select_window.geometry("300x300")
        self.center_popup(select_window)

        tk.Label(select_window, text="Select a sheet name:", bg="#D0D3D4").pack(pady=10)

        # Create a frame to hold the Listbox and Scrollbar
        listbox_frame = tk.Frame(select_window)
        listbox_frame.pack(pady=5, expand=True, fill=tk.BOTH)

        sheet_listbox = tk.Listbox(listbox_frame, selectmode=tk.SINGLE)
        sheet_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        sheet_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=sheet_listbox.yview)

        for name in sheet_names:
            sheet_listbox.insert(tk.END, name)

        # OK Button to confirm the selection
        tk.Button(select_window, text="OK", command=on_select, bg="#FFFFFF").pack(pady=10)

        select_window.wait_window()

        return sheet_name.get()

    def select_spring_file(self):
        self.spring_file_path = filedialog.askopenfilename(title="Select Spring CSV File", filetypes=[("CSV files", "*.csv")])
        if self.spring_file_path:
            self.spring_file_label.config(text=self.spring_file_path)

    def select_fall_file(self):
        self.fall_file_path = filedialog.askopenfilename(title="Select Fall CSV File", filetypes=[("CSV files", "*.csv")])
        if self.fall_file_path:
            self.fall_file_label.config(text=self.fall_file_path)

    def start_generation(self):
        try:
            gen_number = int(self.gen_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Generation number must be an integer.")
            return

        if not self.spring_file_path or not self.fall_file_path:
            messagebox.showerror("Error", "Please select both Spring and Fall CSV files.")
            return

        self.ok_btn.grid_remove()  # Hide the OK button after clicking
        self.back_btn.grid_remove()  # Hide the Back to Main Menu button after clicking
        self.progress_bar.grid()  # Show the progress bar
        self.progress_label.config(text="Generation Started...")
        self.generate_schedule(gen_number)

    def generate_schedule(self, gen_number):
        total_generations = 5  # Example total, you can adjust based on your specific need

        self.progress_bar["value"] = 0
        self.progress_bar["maximum"] = total_generations

        for generation in range(1, total_generations + 1):
            # Simulate generation process
            percentage_complete = int(generation / total_generations * 100)
            self.progress_label.config(text=f"Generation {generation}: [{percentage_complete}%]")
            self.progress_bar["value"] = generation
            self.update_idletasks()

            # Run a generation step (replace this with actual logic)
            algo.runner(gen_number, self.spring_file_path, self.fall_file_path)

        self.progress_label.config(text="Generation Completed!")
        messagebox.showinfo("Success", "Course schedule generated successfully!")

        # Reset progress bar and labels after completion
        self.progress_bar.grid_remove()  # Hide the progress bar again
        self.progress_bar["value"] = 0
        self.spring_file_label.config(text="No file selected")
        self.fall_file_label.config(text="No file selected")
        self.gen_entry.delete(0, tk.END)
        self.progress_label.config(text="")  # Corrected the config method

        self.ok_btn.grid()  # Show the OK button again
        self.back_btn.grid()  # Show the Back to Main Menu button again

    def display_classroom_visualizer(self):
        semester = self.semester_entry.get().lower()
        if semester in ["spring", "fall"]:
            file_path = f'./Excel/Best_Schedule_{semester.capitalize()}.csv'
            visualizer.visualize_schedule(file_path)

    def display_instructor_visualizer(self):
        instructor_name = self.instructor_entry.get()
        semester = self.instructor_semester_entry.get().lower()
        if instructor_name and semester in ["spring", "fall"]:
            file_path = f'./Excel/Best_Schedule_{semester.capitalize()}.csv'
            instructor_visualizer.visualize_instructor_schedule(file_path, instructor_name)

    def get_input(self, prompt):
        input_window = tk.Toplevel(self)
        input_window.title("Input")
        input_window.geometry("300x100")
        self.center_popup(input_window)

        label = tk.Label(input_window, text=prompt, bg="#D0D3D4")
        label.pack(pady=5)

        user_input = tk.StringVar()

        entry = tk.Entry(input_window, textvariable=user_input)
        entry.pack(pady=5)
        entry.focus()

        def close_window():
            input_window.destroy()

        submit_btn = tk.Button(input_window, text="Submit", command=close_window, bg="#FFFFFF")
        submit_btn.pack(pady=5)

        input_window.wait_window()

        return user_input.get()

if __name__ == "__main__":
    app = CourseSchedulerGUI()
    app.mainloop()
