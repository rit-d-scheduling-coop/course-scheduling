#SUMMER 2024
import pandas as pd
import numpy as np
import random
import pygad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Function to load and preprocess the courses data
def load_and_preprocess(file_path, semester):
    courses_df = pd.read_csv(file_path)
    if semester == 'spring':
        columns_to_use = ['Sbjct', 'Crs #', 'Sect#', 'Course Name', ' Instructor', 'Days', 'Start Time.1', 'End Time.1', 'Room #']
        courses = courses_df[columns_to_use].reset_index(drop=True)
        courses.columns = ['Subject', 'Cat#', 'Sect#', 'Course Name', 'Instructor', 'Days', 'Time Start', 'Time End', 'Room #']
    elif semester == 'fall':
        columns_to_use = ['Subject', 'Cat#', 'Sect#', 'Course Name', 'Instructor', 'Days', 'Time Start', 'Time End', 'Room #']
        courses = courses_df[columns_to_use].reset_index(drop=True)
    else:
        raise ValueError("Semester must be either 'spring' or 'fall'")
    
    courses = courses.dropna(subset=['Course Name']).reset_index(drop=True)
    return courses_df, courses

# Load and preprocess the spring and fall course data
spring_courses_df, spring_courses = load_and_preprocess('Spring_2023_Filtered_Corrected.csv', 'spring')
fall_courses_df, fall_courses = load_and_preprocess('Fall_2022_Filtered_Corrected.csv', 'fall')

# Function to decode a chromosome into a schedule
def decode_chromosome(chromosome, courses_cleaned, possible_days, possible_time_slots, possible_lab_days, classrooms):
    schedule = []
    num_courses = len(courses_cleaned)
    for i in range(num_courses):
        # if courses_cleaned.iloc[i]['IsHeader']:
        #     schedule.append(courses_cleaned.iloc[i].drop('IsHeader').to_dict())
        #     continue
        days_index = int(chromosome[3*i] % 2)
        time_slot_index = int(chromosome[3*i + 1] % len(possible_time_slots))
        classroom_index = int(chromosome[3*i + 2] % len(classrooms))
        days = possible_days[days_index]
        time_slot = possible_time_slots[time_slot_index]
        classroom = classrooms[classroom_index]
        course = courses_cleaned.iloc[i]
        # If the course is a lab, schedule it on a different day within the same block
        if 'Lab' in course['Course Name']:
            days = random.choice(possible_lab_days[possible_days[days_index]])
        schedule.append({
            'Subject': course['Subject'],
            'Cat#': course['Cat#'],
            'Sect#': course['Sect#'],
            'Course Name': course['Course Name'],
            'Instructor': course['Instructor'],
            'Days': days,
            'Time Start': time_slot[0],
            'Time End': time_slot[1],
            'Room #': classroom
        })
    return schedule

# Fitness function to evaluate schedules
def fitness_func(ga_instance, solution, solution_idx, courses_cleaned, possible_days, possible_time_slots, possible_lab_days, classrooms):
    schedule = decode_chromosome(solution, courses_cleaned, possible_days, possible_time_slots, possible_lab_days, classrooms)
    fitness = 0
    classroom_schedule = {room: {} for room in classrooms}

    for course in schedule:
        if 'IsHeader' in course and course['IsHeader']:
            continue
        
        # Add fitness points for valid day and time slots
        if course['Days'] in possible_days or course['Days'] in ['M', 'T', 'W', 'R', 'F']:
            fitness += 1
        if course['Time Start'] and course['Time End']:
            fitness += 1

        # Check if Room # is valid (not nan)
        if pd.isna(course['Room #']):
            fitness -= 5  # Penalty for missing room
            continue

        # Check for conflicts and update classroom schedule
        days = course['Days'] if isinstance(course['Days'], list) else [course['Days']]
        for day in days:
            if day not in classroom_schedule[course['Room #']]:
                classroom_schedule[course['Room #']][day] = []
            
            current_slot = (course['Time Start'], course['Time End'])
            for existing_slot in classroom_schedule[course['Room #']][day]:
                if (current_slot[0] < existing_slot[1] and current_slot[1] > existing_slot[0]):
                    fitness -= 10  # Heavy penalty for classroom conflict
            
            classroom_schedule[course['Room #']][day].append(current_slot)

        # Check for instructor conflicts
        for other_course in schedule:
            if course != other_course and course['Instructor'] == other_course['Instructor']:
                if any(day in other_course['Days'] for day in days) and course['Time Start'] == other_course['Time Start']:
                    fitness -= 5  # Penalty for instructor conflict

        # Ensure labs are on a different day within the same block (MWF or TR)
        if 'Lab' in course['Course Name']:
            main_course_days = possible_lab_days.get(course['Days'])
            if main_course_days and not any(day in main_course_days for day in days):
                fitness -= 2

    return fitness

# Function to generate the schedule using genetic algorithm
def generate_schedule(courses_cleaned, semester):
    # Load classroom data
    classrooms_df = pd.read_csv('Excel files/classrooms.csv')
    classrooms = classrooms_df['classroom'].tolist()

    possible_days = ['MWF', 'TR']
    possible_lab_days = {'MWF': ['M', 'W', 'F'], 'TR': ['T', 'R']}
    possible_time_slots = [(f"{hour:02d}:00", f"{hour+1:02d}:00") for hour in range(8, 18)]  # 8 AM to 5 PM slots
    num_courses = len(courses_cleaned)
    
    # Define the genetic algorithm parameters
    gene_space = [{'low': 0, 'high': 2}, {'low': 0, 'high': len(possible_time_slots)}, {'low': 0, 'high': len(classrooms)-1}] * num_courses
    
    def fitness_wrapper(ga_instance, solution, solution_idx):
        return fitness_func(ga_instance, solution, solution_idx, courses_cleaned, possible_days, possible_time_slots, possible_lab_days, classrooms)
    
    ga_instance = pygad.GA(num_generations=100,
                           num_parents_mating=10,
                           fitness_func=fitness_wrapper,
                           sol_per_pop=100,
                           num_genes=num_courses * 3,
                           gene_space=gene_space,
                           parent_selection_type="sss",
                           keep_parents=5,
                           crossover_type="single_point",
                           mutation_type="random",
                           mutation_percent_genes=10)
    
    ga_instance.run()
    
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_schedule = decode_chromosome(solution, courses_cleaned, possible_days, possible_time_slots, possible_lab_days, classrooms)
    best_schedule_df = pd.DataFrame(best_schedule)
    
    output_path = f'Best_Schedule_{semester.capitalize()}.csv'
    best_schedule_df.to_csv(output_path, index=False)
    
    return output_path

def generate_visual_timetable(schedule_df, semester):
    days = ['M', 'T', 'W', 'R', 'F']
    times = [f"{hour:02d}:00" for hour in range(8, 19)]  # 8 AM to 6 PM slots

    # Extract unique classrooms
    classrooms = schedule_df['Room #'].unique()
    classroom_height = 6  # Adjust this value to give more height to each classroom
    box_height = 5  # Adjust this value to make the colored boxes bigger

    fig, ax = plt.subplots(figsize=(14, 10))  # Increased figure size for better readability
    ax.set_title(f'{semester.capitalize()} Timetable')
    ax.set_xlim(0, len(times))
    ax.set_ylim(0, len(classrooms) * len(days) * classroom_height)
    ax.set_xticks(np.arange(len(times)))
    ax.set_xticklabels(times)

    y_labels = []
    y_ticks = []

    for i, room in enumerate(classrooms):
        for j, day in enumerate(days):
            y_labels.append(f'{room} ({day})')
            y_ticks.append((i * len(days) + j) * classroom_height + classroom_height / 2)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    room_day_offset = {f'{room}_{day}': (i * len(days) + j) * classroom_height for i, room in enumerate(classrooms) for j, day in enumerate(days)}

    for idx, course in schedule_df.iterrows():
        day_indices = [days.index(day) for day in course['Days']]
        start_time_index = times.index(course['Time Start'])
        end_time_index = times.index(course['Time End'])
        width = end_time_index - start_time_index
        color = np.random.rand(3,)
        for day_index in day_indices:
            room_day_key = f'{course["Room #"]}_{days[day_index]}'
            y_pos = room_day_offset[room_day_key]
            rect = mpatches.Rectangle((start_time_index, y_pos), width, box_height, color=color, ec='black')
            ax.add_patch(rect)
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width() / 2.0
            cy = ry + rect.get_height() / 2.0
            ax.annotate(f"{course['Course Name']}\n{course['Instructor']}", (cx, cy), color='black', weight='bold', fontsize=8, ha='center', va='center')

    plt.show()

# Generate schedules for spring and fall
spring_schedule_path, fall_schedule_path = generate_schedule(spring_courses, 'spring'), generate_schedule(fall_courses, 'fall')

spring_schedule_path, fall_schedule_path

#(for example) TR might have 1:20hr and then 2/3 hr classes. need to find way to have flexible timing. smaller courses might be "inside" the 2 hour block
# 3 dimensional array time,day,and blocks