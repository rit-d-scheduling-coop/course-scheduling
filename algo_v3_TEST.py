import csv
from datetime import datetime, timedelta
import random
import pygad
import numpy as np

# Global variables
processed_data = None
timeslots = None
classrooms = None
fixed_courses = None
courses_to_schedule = None

def process_csv(file_path):
    result = {'header': []}
    current_department = None

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        
        # Process header
        header = next(csv_reader)
        result['header'] = [header]
        
        for row in csv_reader:
            # Check if this is a department row
            if all(field.strip() == '' for field in row[:4]) and row[4].strip() != '' and all(field.strip() == '' for field in row[5:]):
                current_department = row[4].strip()
                result[current_department] = []
            elif current_department:
                result[current_department].append(row)

    return result

def read_timeslots_csv(file_path):
    timeslots = {}
    
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            day = row['Day']
            start_time = row['Start Time']
            end_time = row['End Time']
            duration = float(row['Duration'])
            
            if day not in timeslots:
                timeslots[day] = []
            
            timeslots[day].append((start_time, end_time, duration))
    
    return timeslots

def read_classrooms_csv(file_path):
    classrooms = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        for row in reader:
            classrooms.append({
                'name': row['classroom'],
                'capacity': int(row['capacity']),
                'type': row['type']
            })
    return classrooms

# Helper function to check if a course is a lab
def is_lab_course(course):
    return 'L' in course[3]

# Helper function to determine required timeslots
def required_timeslots(course):
    hours = int(course[5])
    if is_lab_course(course) or hours in [0, 1]:
        return 1
    return 2

# Helper function to check if a timeslot is valid for a course
def is_valid_timeslot(course, timeslot, classroom):
    students = int(course[7])
    return (not is_lab_course(course) or classroom['type'] == 'L') and students <= classroom['capacity']

def encode_day(day):
    days = list(timeslots.keys())
    return days.index(day)

def encode_timeslot(day, start, end):
    for index, (s, e, _) in enumerate(timeslots[day]):
        if s == start and e == end:
            return index
    raise ValueError(f"Timeslot {start}-{end} not found for day {day}")

def encode_classroom(classroom):
    return next(i for i, c in enumerate(classrooms) if c['name'] == classroom)

def decode_day(day_index):
    days = list(timeslots.keys())
    return days[int(day_index) % len(days)]

def decode_timeslot(day, timeslot_index):
    if day not in timeslots:
        return None, None
    day_timeslots = timeslots[day]
    if timeslot_index < 0 or timeslot_index >= len(day_timeslots):
        return None, None
    return day_timeslots[timeslot_index][:2]  # Return only start and end times

def decode_classroom(classroom_index):
    return classrooms[int(classroom_index) % len(classrooms)]['name']

def create_individual(num_genes):
    individual = []
    days = list(timeslots.keys())
    while len(individual) < num_genes:
        day_index = random.randint(0, len(days) - 1)
        day = days[day_index]
        timeslot_index = random.randint(0, len(timeslots[day]) - 1)
        classroom_index = random.randint(0, len(classrooms) - 1)
        individual.extend([day_index, timeslot_index, classroom_index])
    return individual[:num_genes]  # Ensure the individual is exactly num_genes long

def create_initial_population(sol_per_pop, num_genes):
    return [create_individual(num_genes) for _ in range(sol_per_pop)]

def fitness_func(ga_instance, solution, solution_idx):
    if len(solution) != ga_instance.num_genes:
        print(f"Warning: Solution {solution_idx} has incorrect length {len(solution)}, expected {ga_instance.num_genes}")
        return 0  # Return lowest fitness for invalid solutions

    schedule = []
    idx = 0
    invalid_values = 0
    for course in courses_to_schedule:
        slots_needed = required_timeslots(course)
        course_schedule = []
        for _ in range(slots_needed):
            day_index, timeslot_index, classroom_index = map(int, solution[idx:idx+3])
            day = decode_day(day_index)
            start, end = decode_timeslot(day, timeslot_index)
            classroom = decode_classroom(classroom_index)
            
            if start is None or end is None:
                invalid_values += 1
                course_schedule.append((day, "Invalid", "Invalid", classroom))
            else:
                course_schedule.append((day, start, end, classroom))
            idx += 3
        schedule.append((course, course_schedule))
    
    conflicts = 0
    
    # Check for classroom conflicts
    classroom_usage = {}
    for course, slots in schedule:
        for day, start, end, classroom in slots:
            if start != "Invalid" and end != "Invalid":
                key = (day, start, end, classroom)
                if key in classroom_usage:
                    conflicts += 1
                classroom_usage[key] = course

    # Check for instructor conflicts
    instructor_schedule = {}
    for course, slots in schedule:
        instructor = course[1]
        for day, start, end, _ in slots:
            if start != "Invalid" and end != "Invalid":
                key = (instructor, day, start, end)
                if key in instructor_schedule:
                    conflicts += 1
                instructor_schedule[key] = course

    # Check conflicts with fixed courses
    for fixed_course in fixed_courses:
        day, start, end, classroom = fixed_course[9:13]
        key = (day, start, end, classroom)
        if key in classroom_usage:
            conflicts += 1

    fitness = 1 / (conflicts + invalid_values + 1)  # Higher fitness for fewer conflicts and invalid values
    return fitness

def on_generation(ga_instance):
    print(f"Generation: {ga_instance.generations_completed}")
    print(f"Best Fitness: {ga_instance.best_solution()[1]}")

def custom_mutation_func(offspring, ga_instance):
    for chromosome in offspring:
        for gene_idx in range(len(chromosome)):
            if random.random() < ga_instance.mutation_percent_genes:
                if gene_idx % 3 == 0:  # Day index
                    chromosome[gene_idx] = random.randint(0, len(timeslots) - 1)
                elif gene_idx % 3 == 1:  # Timeslot index
                    day = decode_day(int(chromosome[gene_idx - 1]))
                    chromosome[gene_idx] = random.randint(0, len(timeslots[day]) - 1)
                else:  # Classroom index
                    chromosome[gene_idx] = random.randint(0, len(classrooms) - 1)
    return offspring

def schedule_courses():
    global processed_data, timeslots, classrooms, fixed_courses, courses_to_schedule

    processed_data = process_csv('Excel/Fall_2023_Filtered_Corrected_Updated_v2.csv')
    timeslots = read_timeslots_csv('Excel/timeslots.csv')
    print("Timeslots:", timeslots)
    classrooms = read_classrooms_csv('Excel/classrooms.csv')

    print(f"Number of classrooms: {len(classrooms)}")

    fixed_courses = processed_data['MATH/SCIENCE']
    courses_to_schedule = [course for dept, courses in processed_data.items() 
                           for course in courses if dept != 'MATH/SCIENCE' and dept != 'header']

    num_genes = sum(required_timeslots(course) * 3 for course in courses_to_schedule)
    print(f"Expected number of genes: {num_genes}")

    sol_per_pop = 50  # Number of solutions per population
    num_generations = 1000
    num_parents_mating = 10

    initial_population = create_initial_population(sol_per_pop, num_genes)
    
    # Check if all individuals have the correct length
    for i, individual in enumerate(initial_population):
        if len(individual) != num_genes:
            print(f"Warning: Individual {i} has incorrect length {len(individual)}, expected {num_genes}")
            individual.extend([0] * (num_genes - len(individual)))
            individual = individual[:num_genes]

    max_timeslot = max(len(slots) for slots in timeslots.values())
    init_range_high = max(len(timeslots), max_timeslot, len(classrooms))

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           num_genes=num_genes,
                           sol_per_pop=sol_per_pop,
                           init_range_low=0,
                           init_range_high=init_range_high,
                           parent_selection_type="sss",
                           keep_parents=1,
                           crossover_type="single_point",
                           mutation_type=custom_mutation_func,
                           mutation_percent_genes=10,
                           on_generation=on_generation,
                           initial_population=initial_population)

    ga_instance.run()

    solution, solution_fitness, _ = ga_instance.best_solution()
    print(f"Best solution fitness: {solution_fitness}")

    return solution

# Generate CSV output (you'll need to implement this based on your specific needs)
def generate_csv_output(best_schedule):
    output_file = 'Excel/Best_Schedule_Fall.csv'
    header = processed_data['header'][0]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        schedule_index = 0
        for department, courses in processed_data.items():
            if department == 'header':
                continue
            
            writer.writerow([''] * 4 + [department] + [''] * 7)
            
            for course in courses:
                if department == 'MATH/SCIENCE':
                    writer.writerow(course)
                else:
                    slots_needed = required_timeslots(course)
                    for _ in range(slots_needed):
                        day_index, timeslot_index, classroom_index = best_schedule[schedule_index:schedule_index+3]
                        schedule_index += 3
                        
                        day = decode_day(int(day_index))
                        start, end = decode_timeslot(day, int(timeslot_index))
                        classroom = decode_classroom(int(classroom_index))
                        
                        if start is None or end is None:
                            start, end = "Invalid", "Invalid"
                        
                        classroom_type = next((c['type'] for c in classrooms if c['name'] == classroom), 'Unknown')
                        
                        new_row = course[:8] + [day, start, end, classroom_type, classroom]
                        writer.writerow(new_row)

    print(f"Schedule has been written to {output_file}")


# Run the algorithm
best_schedule = schedule_courses()

generate_csv_output(best_schedule)