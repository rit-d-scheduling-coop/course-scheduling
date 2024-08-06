import pandas as pd
import numpy as np
import random
import pygad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add the timeslots dictionary
timeslots = {
    "MWF": [
        ("08:59", "09:54", 1.0),
        ("10:04", "10:59", 1.0),
        ("11:09", "12:04", 1.0)
    ],
    "MW": [
        ("13:04", "14:24", 1.25),
        ("14:34", "15:54", 1.25),
        ("16:04", "17:24", 1.25)
    ],
    "TR": [
        ("08:59", "10:19", 1.5),
        ("10:29", "11:49", 1.25),
        ("11:59", "13:19", 1.25),
        ("13:29", "14:49", 1.25),
        ("14:59", "16:19", 1.25),
        ("16:29", "17:49", 1.25)
    ]
}

# Function to load and preprocess the courses data
def load_and_preprocess(file_path):
    courses_df = pd.read_csv(file_path)
    columns_to_use = ['Class #', 'Subject', 'Cat#', 'Sect#', 'Course Name', 'Hrs', 'Instructor', 'Enr Cap', 'Days', 'Time Start', 'Time End', 'Room #']
    courses = courses_df[columns_to_use].reset_index(drop=True)
    
    # Fill NaN values in 'Course Name' with a placeholder
    courses['Course Name'] = courses['Course Name'].fillna('Unknown Course')
    
    # Mark rows where only the 'Course Name' column is filled and others are missing (department headers)
    courses['IsHeader'] = courses.apply(lambda row: pd.notna(row['Course Name']) and row[['Subject', 'Cat#', 'Sect#', 'Instructor', 'Days', 'Time Start', 'Time End', 'Room #']].isnull().all(), axis=1)
    
    courses = courses.reset_index(drop=True)
    return courses_df, courses

# Function to load prerequisites data
def load_prerequisites(file_path):
    prerequisites_df = pd.read_csv(file_path)
    prerequisites = {}
    for _, row in prerequisites_df.iterrows():
        course = row['Course']
        prerequisite = row['Prerequisite']
        if course not in prerequisites:
            prerequisites[course] = []
        if pd.notna(prerequisite):
            prerequisites[course].append(prerequisite)
    return prerequisites

def load_classrooms(file_path):
    classrooms_df = pd.read_csv(file_path, skipinitialspace=True)
    # Separate classrooms into regular and lab types
    regular_classrooms = classrooms_df[classrooms_df['type'] != 'L']['classroom'].tolist()
    lab_classrooms = classrooms_df[classrooms_df['type'] == 'L']['classroom'].tolist()
    return regular_classrooms, lab_classrooms

# Function to get all prerequisites (direct and transitive) for a course
def get_all_prerequisites(course, prerequisites, all_prereqs=None):
    if all_prereqs is None:
        all_prereqs = set()
    if course in prerequisites:
        for prereq in prerequisites[course]:
            if prereq not in all_prereqs:
                all_prereqs.add(prereq)
                get_all_prerequisites(prereq, prerequisites, all_prereqs)
    return all_prereqs

# Load and preprocess the spring and fall course data
spring_courses_df, spring_courses = load_and_preprocess('Spring_2024_Filtered_Corrected_Updated_v4.csv')
fall_courses_df, fall_courses = load_and_preprocess('Fall_2023_Filtered_Corrected_Updated_v4.csv')

# Function to decode a chromosome into a schedule
def decode_chromosome(chromosome, courses_cleaned, possible_days, timeslots, possible_lab_days, regular_classrooms, lab_classrooms):
    schedule = []
    num_courses = len(courses_cleaned)
    fixed_subjects = {'ACSC', 'BIOG', 'BIOL', 'CHMG', 'MATH', 'PHYS'}

    for i in range(num_courses):
        course = courses_cleaned.iloc[i]
        if course['IsHeader']:
            schedule.append(course.drop('IsHeader').to_dict())
            continue

        if course['Subject'] in fixed_subjects:
            schedule.append(course.drop('IsHeader').to_dict())
            continue

        days_index = int(chromosome[3*i] % len(possible_days))
        days = possible_days[days_index]
        time_slot_index = int(chromosome[3*i + 1] % len(timeslots[days]))
        time_slot = timeslots[days][time_slot_index]
        
        # Determine if it's a lab course based on Sect# pattern
        is_lab = 'L' in str(course['Sect#']) and str(course['Sect#']).split('L')[1].isdigit()
        
        # Choose classroom based on whether it's a lab course or not
        if is_lab:
            classroom_index = int(chromosome[3*i + 2] % len(lab_classrooms))
            classroom = lab_classrooms[classroom_index]
        else:
            classroom_index = int(chromosome[3*i + 2] % len(regular_classrooms))
            classroom = regular_classrooms[classroom_index]
        
        if is_lab:
            days = random.choice(possible_lab_days[days])
        
        schedule.append({
            'Class #': course['Class #'],
            'Subject': course['Subject'],
            'Cat#': course['Cat#'],
            'Sect#': course['Sect#'],
            'Course Name': course['Course Name'],
            'Hrs': course['Hrs'],
            'Instructor': course['Instructor'],
            'Enr Cap': course['Enr Cap'],
            'Days': days,
            'Time Start': time_slot[0],
            'Time End': time_slot[1],
            'Room #': classroom
        })

    return schedule

# Fitness function to evaluate schedules
def fitness_func(ga_instance, solution, solution_idx, courses_cleaned, possible_days, timeslots, possible_lab_days, regular_classrooms, lab_classrooms):
    schedule = decode_chromosome(solution, courses_cleaned, possible_days, timeslots, possible_lab_days, regular_classrooms, lab_classrooms)

    fitness = 0
    classroom_schedule = {room: {} for room in regular_classrooms + lab_classrooms}
    instructor_schedule = {}

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

        # Check for classroom conflicts and update classroom schedule
        days = course['Days'] if isinstance(course['Days'], list) else [course['Days']]
        for day in days:
            if course['Room #'] not in classroom_schedule:
                classroom_schedule[course['Room #']] = {}
            if day not in classroom_schedule[course['Room #']]:
                classroom_schedule[course['Room #']][day] = []
            
            current_slot = (course['Time Start'], course['Time End'])
            if any(not isinstance(t, str) for t in current_slot):
                continue  # Skip invalid time entries

            for existing_slot in classroom_schedule[course['Room #']][day]:
                if (current_slot[0] < existing_slot[1] and current_slot[1] > existing_slot[0]):
                    fitness -= 10  # Heavy penalty for classroom conflict
            
            classroom_schedule[course['Room #']][day].append(current_slot)

        # Check for instructor conflicts and update instructor schedule
        instructor = course['Instructor']
        if instructor == 'TBD':
            continue

        if instructor not in instructor_schedule:
            instructor_schedule[instructor] = {}

        for day in days:
            if day not in instructor_schedule[instructor]:
                instructor_schedule[instructor][day] = []

            current_slot = (course['Time Start'], course['Time End'])
            if any(not isinstance(t, str) for t in current_slot):
                continue  # Skip invalid time entries

            for existing_slot in instructor_schedule[instructor][day]:
                if (current_slot[0] < existing_slot[1] and current_slot[1] > existing_slot[0]):
                    fitness -= 5  # Penalty for instructor conflict
            
            instructor_schedule[instructor][day].append(current_slot)

        # Ensure labs are on a different day within the same block (MWF or TR)
        if 'Lab' in course['Course Name']:
            main_course_days = possible_lab_days.get(course['Days'])
            if main_course_days and not any(day in main_course_days for day in days):
                fitness -= 2

    return fitness

# Function to generate the schedule using genetic algorithm
def generate_schedule(courses_cleaned, semester):
    # Load classrooms from the CSV file
    regular_classrooms, lab_classrooms = load_classrooms('excel/classrooms.csv')

    possible_days = list(timeslots.keys())
    possible_lab_days = {'MWF': ['M', 'W', 'F'], 'TR': ['T', 'R'], 'MW': ['M', 'W']}
    num_courses = len(courses_cleaned)
    
    # Define the genetic algorithm parameters
    gene_space = []
    for _ in range(num_courses):
        gene_space.extend([
            {'low': 0, 'high': len(possible_days)},
            {'low': 0, 'high': max(len(slots) for slots in timeslots.values())},
            {'low': 0, 'high': max(len(regular_classrooms), len(lab_classrooms)) - 1}
        ])
    
    def fitness_wrapper(ga_instance, solution, solution_idx):
        return fitness_func(ga_instance, solution, solution_idx, courses_cleaned, possible_days, timeslots, possible_lab_days, regular_classrooms, lab_classrooms)
    
    ga_instance = pygad.GA(num_generations=20,
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
    best_schedule = decode_chromosome(solution, courses_cleaned, possible_days, timeslots, possible_lab_days, regular_classrooms, lab_classrooms)
    best_schedule_df = pd.DataFrame(best_schedule)
    
    # Ensure all columns are present in the output
    columns_order = ['Class #', 'Subject', 'Cat#', 'Sect#', 'Course Name', 'Hrs', 'Instructor', 'Enr Cap', 'Days', 'Time Start', 'Time End', 'Room #']
    best_schedule_df = best_schedule_df.reindex(columns=columns_order)
    
    output_path = f'Excel/Best_Schedule_{semester.capitalize()}.csv'
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
            y_ticks.append((i * len(days) + j + 0.5) * classroom_height)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    room_day_offset = {}
    for i, room in enumerate(classrooms):
        for j, day in enumerate(days):
            room_day_key = f'{room}_{day}'
            room_day_offset[room_day_key] = (i * len(days) + j) * classroom_height

    for _, course in schedule_df.iterrows():
        if 'IsHeader' in course and course['IsHeader']:
            continue
        day_indices = [days.index(d) for d in course['Days']]
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

# Update the generate_schedule function calls
spring_schedule_path = generate_schedule(spring_courses, 'spring')
fall_schedule_path = generate_schedule(fall_courses, 'fall')

# Print the paths to the generated schedules
print(spring_schedule_path, fall_schedule_path)
