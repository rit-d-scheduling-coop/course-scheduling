import pandas as pd
import numpy as np
import random
import pygad

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
def decode_chromosome(chromosome, courses_cleaned, possible_days, possible_time_slots, possible_lab_days):
    schedule = []
    num_courses = len(courses_cleaned)
    for i in range(num_courses):
        days_index = int(chromosome[2*i] % 2)
        time_slot_index = int(chromosome[2*i + 1] % len(possible_time_slots))
        days = possible_days[days_index]
        time_slot = possible_time_slots[time_slot_index]
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
            'Room #': course['Room #']
        })
    return schedule

# Fitness function to evaluate schedules
def fitness_func(ga_instance, solution, solution_idx, courses_cleaned, possible_days, possible_time_slots, possible_lab_days):
    schedule = decode_chromosome(solution, courses_cleaned, possible_days, possible_time_slots, possible_lab_days)
    fitness = 0
    for course in schedule:
        # Add fitness points for valid day and time slots
        if course['Days'] in possible_days or course['Days'] in ['M', 'T', 'W', 'R', 'F']:
            fitness += 1
        if course['Time Start'] and course['Time End']:
            fitness += 1
        # Check for overlaps
        for other_course in schedule:
            if course != other_course and course['Instructor'] == other_course['Instructor']:
                if course['Days'] == other_course['Days'] and course['Time Start'] == other_course['Time Start']:
                    fitness -= 1
            # Ensure labs are on a different day within the same block (MWF or TR)
            if 'Lab' in course['Course Name']:
                main_course_days = possible_lab_days.get(other_course['Days'])
                if main_course_days and course['Days'] not in main_course_days:
                    fitness -= 1
    return fitness

# Function to generate the schedule using genetic algorithm
def generate_schedule(courses_df, courses_cleaned, semester):
    # Define possible days and time slots
    possible_days = ['MWF', 'TR']
    possible_lab_days = {'MWF': ['M', 'W', 'F'], 'TR': ['T', 'R']}
    possible_time_slots = [(f"{hour:02d}:00", f"{hour+1:02d}:00") for hour in range(8, 18)]  # 8 AM to 5 PM slots
    num_courses = len(courses_cleaned)
    
    # Define the genetic algorithm parameters
    gene_space = [{'low': 0, 'high': 2}, {'low': 0, 'high': len(possible_time_slots)}] * num_courses
    
    def fitness_wrapper(ga_instance, solution, solution_idx):
        return fitness_func(ga_instance, solution, solution_idx, courses_cleaned, possible_days, possible_time_slots, possible_lab_days)
    
    ga_instance = pygad.GA(num_generations=10,
                           num_parents_mating=10,
                           fitness_func=fitness_wrapper,
                           sol_per_pop=50,
                           num_genes=num_courses * 2,
                           gene_space=gene_space,
                           parent_selection_type="sss",
                           keep_parents=5,
                           crossover_type="single_point",
                           mutation_type="random",
                           mutation_percent_genes=10)
    
    # Run the genetic algorithm
    ga_instance.run()
    
    # Get the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_schedule = decode_chromosome(solution, courses_cleaned, possible_days, possible_time_slots, possible_lab_days)
    best_schedule_df = pd.DataFrame(best_schedule)
    
    # Merge with original dataframe to keep department headers
   # department_headers = courses_df[courses_df['Subject'].isna()]
    final_schedule_df = pd.concat([best_schedule_df], ignore_index=True)
    
    # Save the best schedule to a CSV file
    output_path = f'./Best_Schedule_{semester.capitalize()}.csv'
    final_schedule_df.to_csv(output_path, index=False)
    
    print(f"The best schedule for {semester} has been saved to {output_path}")

# Generate schedules for spring and fall
generate_schedule(spring_courses_df, spring_courses, 'spring')
generate_schedule(fall_courses_df, fall_courses, 'fall')
