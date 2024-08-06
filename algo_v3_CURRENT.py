import pandas as pd
import numpy as np
import random
import pygad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add the timeslots dictionary
timeslots = {
    "M": [
        ("08:59", "09:54", 1.0),
        ("10:04", "10:59", 1.0),
        ("11:09", "12:04", 1.0),
        ("13:04", "14:24", 1.25),
        ("14:34", "15:54", 1.25),
        ("16:04", "17:24", 1.25)
    ],
    "W": [
        ("08:59", "09:54", 1.0),
        ("10:04", "10:59", 1.0),
        ("11:09", "12:04", 1.0),
        ("13:04", "14:24", 1.25),
        ("14:34", "15:54", 1.25),
        ("16:04", "17:24", 1.25)
    ],
    "F": [
        ("08:59", "09:54", 1.0),
        ("10:04", "10:59", 1.0),
        ("11:09", "12:04", 1.0)
    ],
    "T": [
        ("08:59", "10:19", 1.5),
        ("10:29", "11:49", 1.25),
        ("11:59", "13:19", 1.25),
        ("13:29", "14:49", 1.25),
        ("14:59", "16:19", 1.25),
        ("16:29", "17:49", 1.25)
    ],
    "R": [
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
    
    # Convert 'Enr Cap' to integer, replacing non-numeric values with 0
    courses['Enr Cap'] = pd.to_numeric(courses['Enr Cap'], errors='coerce').fillna(0).astype(int)
    
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
    # Convert capacity to integer and create dictionaries
    classrooms_df['capacity'] = classrooms_df['capacity'].astype(int)
    regular_classrooms = classrooms_df[classrooms_df['type'] != 'L'][['classroom', 'capacity']].to_dict('records')
    lab_classrooms = classrooms_df[classrooms_df['type'] == 'L'][['classroom', 'capacity']].to_dict('records')
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

# Function to separate fixed and non-fixed courses
def separate_courses(courses):
    fixed_subjects = {'ACSC', 'BIOG', 'BIOL', 'CHMG', 'MATH', 'PHYS', 'STAT'}
    fixed_courses = courses[courses['Subject'].isin(fixed_subjects)]
    non_fixed_courses = courses[~courses['Subject'].isin(fixed_subjects)]
    return fixed_courses, non_fixed_courses

# Modified decode_chromosome function
def decode_chromosome(chromosome, courses_cleaned, possible_days, timeslots, possible_lab_days, regular_classrooms, lab_classrooms):
    schedule = []
    num_courses = len(courses_cleaned)

    for i in range(num_courses):
        course = courses_cleaned.iloc[i]
        if course['IsHeader']:
            schedule.append(course.drop('IsHeader').to_dict())
            continue

        # Determine if it's a lab course based on Sect# pattern
        is_lab = 'L' in str(course['Sect#']) and str(course['Sect#']).split('L')[1].isdigit()

        if is_lab:
            # For lab courses, assign the day removed from the lecture
            lecture_course = courses_cleaned.iloc[i-1]  # Assuming the lecture is always right before the lab
            lecture_days = schedule[-1]['Days']  # Get the days assigned to the lecture
            if lecture_days == 'MW':
                assigned_day = 'F'
            elif lecture_days == 'MF':
                assigned_day = 'W'
            elif lecture_days == 'WF':
                assigned_day = 'M'
            elif lecture_days == 'T':
                assigned_day = 'R'
            else:  # lecture_days == 'R'
                assigned_day = 'T'
            day_slots = timeslots[assigned_day]
        else:
            # For non-lab courses, assign MWF or TR, then remove one day if it has a lab
            days_index = int(chromosome[3*i] % 2)  # 0 for MWF, 1 for TR
            if days_index == 0:
                if i+1 < num_courses and 'L' in str(courses_cleaned.iloc[i+1]['Sect#']):
                    # This lecture has a lab, remove one day
                    day_remove = int(chromosome[3*i+1] % 3)  # 0 for M, 1 for W, 2 for F
                    if day_remove == 0:
                        assigned_days = "WF"
                    elif day_remove == 1:
                        assigned_days = "MF"
                    else:
                        assigned_days = "MW"
                else:
                    assigned_days = "MWF"
                day_slots = timeslots['M']
            else:
                if i+1 < num_courses and 'L' in str(courses_cleaned.iloc[i+1]['Sect#']):
                    # This lecture has a lab, remove one day
                    assigned_days = "T" if int(chromosome[3*i+1] % 2) == 0 else "R"
                else:
                    assigned_days = "TR"
                day_slots = timeslots['T']

        time_slot_index = int(chromosome[3*i + 1] % len(day_slots))

        hrs = course['Hrs']
        if pd.isna(hrs):
            hrs = 3  # Default to 3 hours if not specified

        # Assign exactly two consecutive timeslots for courses with 'Hrs' > 3
        if hrs > 3:
            if len(day_slots) > 1:
                slot_index = time_slot_index % (len(day_slots) - 1)
                start_time, _, _ = day_slots[slot_index]
                _, end_time, _ = day_slots[slot_index + 1]
            else:
                # If there's only one slot available, use it
                start_time, end_time, _ = day_slots[0]
        else:
            slot_index = time_slot_index % len(day_slots)
            start_time, end_time, _ = day_slots[slot_index]

        # Choose classroom
        enr_cap = int(course['Enr Cap'])
        if is_lab:
            suitable_classrooms = [room for room in lab_classrooms if room['capacity'] >= enr_cap]
            if not suitable_classrooms:
                suitable_classrooms = lab_classrooms
        else:
            suitable_classrooms = [room for room in regular_classrooms if room['capacity'] >= enr_cap]
            if not suitable_classrooms:
                suitable_classrooms = regular_classrooms
        
        classroom_index = int(chromosome[3*i + 2] % len(suitable_classrooms))
        classroom = suitable_classrooms[classroom_index]['classroom']
        capacity = suitable_classrooms[classroom_index]['capacity']
        
        schedule.append({
            'Class #': course['Class #'],
            'Subject': course['Subject'],
            'Cat#': course['Cat#'],
            'Sect#': course['Sect#'],
            'Course Name': course['Course Name'],
            'Hrs': hrs,
            'Instructor': course['Instructor'],
            'Enr Cap': enr_cap,
            'Days': assigned_day if is_lab else assigned_days,
            'Time Start': start_time,
            'Time End': end_time,
            'Room #': classroom,
            'Room Capacity': capacity
        })

    return schedule

# Modified fitness function
def fitness_func(ga_instance, solution, solution_idx, courses_cleaned, possible_days, timeslots, possible_lab_days, regular_classrooms, lab_classrooms):
    schedule = decode_chromosome(solution, courses_cleaned, possible_days, timeslots, possible_lab_days, regular_classrooms, lab_classrooms)

    fitness = 0
    classroom_schedule = {room['classroom']: {} for room in regular_classrooms + lab_classrooms}
    instructor_schedule = {}
    day_distribution = {'M': 0, 'T': 0, 'W': 0, 'R': 0, 'F': 0, 'MWF': 0, 'TR': 0}

    for course in schedule:
        if 'IsHeader' in course and course['IsHeader']:
            continue
        
        # Add fitness points for valid day and time slots
        if course['Days'] in ['M', 'T', 'W', 'R', 'F', 'MWF', 'TR']:
            fitness += 1
            day_distribution[course['Days']] += 1
        if course['Time Start'] and course['Time End']:
            fitness += 1

        # Check if Room # is valid (not nan)
        if pd.isna(course['Room #']):
            fitness -= 5  # Penalty for missing room
            continue

        # Penalize courses with > 3 hours that only have one timeslot
        hrs = course['Hrs']
        if pd.notna(hrs) and hrs > 3:
            day_slots = timeslots['M'] if course['Days'] in ['M', 'W', 'F', 'MWF'] else timeslots['T']
            assigned_slots = sum(1 for slot in day_slots if course['Time Start'] <= slot[0] < course['Time End'])
            if assigned_slots < 2:
                fitness -= 10  # Heavy penalty for not assigning two slots to a >3 hour course

        # Check for classroom conflicts and update classroom schedule
        days = course['Days']
        for day in days:
            if course['Room #'] not in classroom_schedule:
                classroom_schedule[course['Room #']] = {}
            if day not in classroom_schedule[course['Room #']]:
                classroom_schedule[course['Room #']][day] = []
            
            current_slot = (course['Time Start'], course['Time End'])
            if any(not isinstance(t, str) for t in current_slot):
                continue  # Skip invalid time entries

            # Check for conflicts across all timeslots the course occupies
            conflict = False
            for existing_slot in classroom_schedule[course['Room #']][day]:
                if (current_slot[0] < existing_slot[1] and current_slot[1] > existing_slot[0]):
                    conflict = True
                    break
            
            if conflict:
                fitness -= 10  # Heavy penalty for classroom conflict
            else:
                classroom_schedule[course['Room #']][day].append(current_slot)

        # Check for instructor conflicts and update instructor schedule
        instructor = course['Instructor']
        if instructor != 'TBD':
            if instructor not in instructor_schedule:
                instructor_schedule[instructor] = {}

            for day in days:
                if day not in instructor_schedule[instructor]:
                    instructor_schedule[instructor][day] = []

                current_slot = (course['Time Start'], course['Time End'])
                if any(not isinstance(t, str) for t in current_slot):
                    continue  # Skip invalid time entries

                conflict = False
                for existing_slot in instructor_schedule[instructor][day]:
                    if (current_slot[0] < existing_slot[1] and current_slot[1] > existing_slot[0]):
                        conflict = True
                        break
                
                if conflict:
                    fitness -= 5  # Penalty for instructor conflict
                else:
                    instructor_schedule[instructor][day].append(current_slot)

        # Check if the assigned time slots match the required hours
        if pd.notna(hrs):
            day_slots = timeslots['M'] if course['Days'] == 'MWF' else timeslots['T']
            assigned_slots = sum(1 for slot in day_slots if course['Time Start'] <= slot[0] < course['Time End'])
            total_assigned_hours = assigned_slots * 1.5 * (3 if course['Days'] == 'MWF' else 2)
            
            if abs(total_assigned_hours - hrs) <= 0.5:  # Allow for small discrepancies
                fitness += 2
            else:
                fitness -= 2  # Penalty for mismatched hours

    # Encourage balanced distribution across MWF and TR
    mwf_count = day_distribution['MWF'] + day_distribution['M'] + day_distribution['W'] + day_distribution['F']
    tr_count = day_distribution['TR'] + day_distribution['T'] + day_distribution['R']
    total_courses = mwf_count + tr_count
    if total_courses > 0:
        balance_ratio = min(mwf_count, tr_count) / total_courses
    else:
        balance_ratio = 0

    balance_score = balance_ratio * 50  # Increased from 10 to 50

    # Add an additional penalty for large imbalances
    imbalance = abs(mwf_count - tr_count)
    if imbalance > total_courses * 0.1:  # If imbalance is more than 10% of total courses
        balance_score -= (imbalance - total_courses * 0.1) * 2  # Additional penalty

    fitness += balance_score

    return fitness

# Modified generate_schedule function
def generate_schedule(courses_cleaned, semester, num_generations):
    # Load classrooms from the CSV file
    regular_classrooms, lab_classrooms = load_classrooms('excel/classrooms.csv')

    # Separate fixed and non-fixed courses
    fixed_courses, non_fixed_courses = separate_courses(courses_cleaned)

    possible_days = ['MWF', 'TR']
    possible_lab_days = {'MWF': ['M', 'W', 'F'], 'TR': ['T', 'R']}
    num_courses = len(non_fixed_courses)
    
    # Define the genetic algorithm parameters
    gene_space = []
    for _ in range(num_courses):
        gene_space.extend([
            {'low': 0, 'high': len(possible_days)},
            {'low': 0, 'high': max(len(slots) for slots in timeslots.values())},
            {'low': 0, 'high': max(len(regular_classrooms), len(lab_classrooms)) - 1}
        ])
    
    def fitness_wrapper(ga_instance, solution, solution_idx):
        return fitness_func(ga_instance, solution, solution_idx, non_fixed_courses, possible_days, timeslots, possible_lab_days, regular_classrooms, lab_classrooms)
    
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=20,  # Increased from 10
                           fitness_func=fitness_wrapper,
                           sol_per_pop=200,  # Increased from 100
                           num_genes=num_courses * 3,
                           gene_space=gene_space,
                           parent_selection_type="tournament",
                           K_tournament=5,
                           keep_parents=10,  # Increased from 5
                           crossover_type="two_points",
                           mutation_type="random",
                           mutation_percent_genes=15)  # Increased from 10
    
    ga_instance.run()
    
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_schedule = decode_chromosome(solution, non_fixed_courses, possible_days, timeslots, possible_lab_days, regular_classrooms, lab_classrooms)
    
    # Combine fixed courses with the optimized non-fixed courses
    combined_schedule = fixed_courses.to_dict('records') + best_schedule
    best_schedule_df = pd.DataFrame(combined_schedule)
    
    # Ensure all columns are present in the output
    columns_order = ['Class #', 'Subject', 'Cat#', 'Sect#', 'Course Name', 'Hrs', 'Instructor', 'Enr Cap', 'Days', 'Time Start', 'Time End', 'Room #']
    best_schedule_df = best_schedule_df.reindex(columns=columns_order)
    
    output_path = f'Excel/Best_Schedule_{semester.capitalize()}.csv'
    best_schedule_df.to_csv(output_path, index=False)
    
    return output_path

# Load and preprocess the spring and fall course data
spring_courses_df, spring_courses = load_and_preprocess('Spring_2024_Filtered_Corrected_Updated_v4.csv')
fall_courses_df, fall_courses = load_and_preprocess('Fall_2023_Filtered_Corrected_Updated_v4.csv')

# Generate schedule
spring_schedule_path = generate_schedule(spring_courses, 'spring', 200)
fall_schedule_path = generate_schedule(fall_courses, 'fall', 200)

# Print the paths to the generated schedules
print(spring_schedule_path, fall_schedule_path)