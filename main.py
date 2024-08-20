#!/usr/bin/env python3

import os
import algo_v4_CURRENT as algo
import cleaning_MOH_v2 as cleaning
import visualizer
import instructor_visualizer
import colorama
from colorama import Fore, Back, Style

colorama.init(autoreset=True)

class Color:
    RED = Fore.RED
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    BLUE = Fore.BLUE
    MAGENTA = Fore.MAGENTA
    CYAN = Fore.CYAN
    WHITE = Fore.WHITE
    END = Fore.RESET

class Style:
    BOLD = Style.BRIGHT
    END = Style.RESET_ALL

def clear_screen():
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For macOS and Linux
    else:
        _ = os.system('clear')

def colored_input(prompt):
    return input(Color.CYAN + Style.BOLD + prompt + Color.END + Style.END)

def display_title_screen():
    clear_screen()
    print(Color.GREEN + Style.BOLD + '''
 _____                              _____      _              _       _           
/  __ \                            /  ___|    | |            | |     | |          
| /  \/ ___  _   _ _ __ ___  ___   \ `--.  ___| |__   ___  __| |_   _| | ___ _ __ 
| |    / _ \| | | | '__/ __|/ _ \   `--. \/ __| '_ \ / _ \/ _` | | | | |/ _ \ '__|
| \__/\ (_) | |_| | |  \__ \  __/  /\__/ / (__| | | |  __/ (_| | |_| | |  __/ |   
 \____/\___/ \__,_|_|  |___/\___|  \____/ \___|_| |_|\___|\__,_|\__,_|_|\___|_|   
''' + Color.END + Style.END)

    print(Color.YELLOW + '''
Welcome to the Course Scheduler!

This software helps you manage course schedules, clean input data, generate optimal schedules,
and visualize classroom and instructor schedules. Follow the prompts to use the various features.

Options:
1. Clean user input Excel file
2. Generate course schedule
3. Display classroom visualizer
4. Display instructor visualizer
5. Exit
''' + Color.END)

def clean_excel_file():
    file_path = colored_input("Enter the path to your Excel file: ")
    sheet_name = colored_input("Enter the name of the sheet: ")
    cleaning.process_and_save_excel_file(file_path, sheet_name)
    print(Color.GREEN + "Excel file cleaned successfully." + Color.END)

def generate_schedule():
    gen_number = int(colored_input("Enter the generation number: "))
    spring_path = colored_input("Enter the path to the Spring Excel file: ")
    fall_path = colored_input("Enter the path to the Fall Excel file: ")
    spring_schedule, fall_schedule = algo.runner(gen_number, spring_path, fall_path)
    print(Color.GREEN + f"Spring schedule generated: {spring_schedule}" + Color.END)
    print(Color.GREEN + f"Fall schedule generated: {fall_schedule}" + Color.END)

def display_classroom_visualizer():
    semester = colored_input("Enter the semester (spring/fall): ").lower()
    file_path = f'./Excel/Best_Schedule_{semester.capitalize()}.csv'
    visualizer.visualize_schedule(file_path)

def display_instructor_visualizer():
    instructor_name = colored_input("Enter the instructor name: ")
    semester = colored_input("Enter the semester (spring/fall): ").lower()
    file_path = f'./Excel/Best_Schedule_{semester.capitalize()}.csv'
    instructor_visualizer.visualize_instructor_schedule(file_path, instructor_name)

while True:
    display_title_screen()
    choice = colored_input("Enter your choice (1-5): ")
    if choice == '1':
        clean_excel_file()
    elif choice == '2':
        generate_schedule()
    elif choice == '3':
        display_classroom_visualizer()
    elif choice == '4':
        display_instructor_visualizer()
    elif choice == '5':
        print(Color.MAGENTA + "Thank you for using the Course Scheduler. Goodbye!" + Color.END)
        break
    else:
        print(Color.RED + "Invalid choice. Please try again." + Color.END)
    
    input(Color.YELLOW + "\nPress Enter to continue..." + Color.END)