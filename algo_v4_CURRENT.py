import pandas as pd
import numpy as np
import random
import pygad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import concurrent.futures

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
    columns_to_use = ['Class #', 'Subject', 'Cat#', 'Sect#', 'Course Name', 'Hrs', 'Instructor', 'Enr Cap', 'Days', 'Time Start', 'Time End', 'Room #', 'Yr Level/ Reqrmt']
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
    classroom_schedule = {room['classroom']: {day: [] for day in 'MTWRF'} for room in regular_classrooms + lab_classrooms}
    instructor_schedule = {}
    timeslot_usage = {room['classroom']: {day: {slot: 0 for slot in range(len(timeslots[day]))} for day in 'MTWRF'} for room in regular_classrooms + lab_classrooms}
    lab_classroom_usage = {room['classroom']: 0 for room in lab_classrooms}

    for i in range(num_courses):
        course = courses_cleaned.iloc[i]
        if course['IsHeader']:
            schedule.append(course.drop('IsHeader').to_dict())
            continue

        is_lab = 'L' in str(course['Sect#'])
        hrs = course['Hrs'] if pd.notna(course['Hrs']) else 3

        # Assign days and time slots based on the rules
        if is_lab:
            lecture_index = i - 1
            while lecture_index >= 0 and 'L' in str(courses_cleaned.iloc[lecture_index]['Sect#']):
                lecture_index -= 1
            if lecture_index >= 0:
                lecture_days = schedule[lecture_index]['Days']
                assigned_day = 'F' if lecture_days == 'MW' else ('W' if lecture_days == 'MF' else 'M' if lecture_days == 'WF' else 'R' if lecture_days == 'T' else 'T')
            else:
                assigned_day = random.choice('MTWRF')
            assigned_days = assigned_day  # Ensure assigned_days is a string
        else:
            if hrs == 1 or hrs == 0:
                assigned_days = random.choice('MTWRF')
                required_slots = 1
            elif hrs == 2:
                if int(chromosome[3*i] % 2) == 0:  # Choose between single day or two days
                    assigned_days = random.choice('MTWRF')
                    required_slots = 2
                else:
                    assigned_days = random.choice(['MW', 'WF', 'MF', 'TR'])
                    required_slots = 1
            elif hrs == 3:
                assigned_days = random.choice(['MW', 'WF', 'MF', 'TR'])
                required_slots = 1
            elif hrs == 6:
                assigned_days = 'TR' if int(chromosome[3*i] % 2) == 0 else 'MWF'
                required_slots = 2
            else:
                # Default case (for any other hour value)
                assigned_days = random.choice(['MWF', 'TR'])
                required_slots = 1

        # Find available time slots
        available_slots = []
        common_slots = set(range(len(timeslots[assigned_days[0]])))
        for day in assigned_days[1:]:
            common_slots &= set(range(len(timeslots[day])))
        
        for slot_index in common_slots:
            if slot_index + required_slots > min(len(timeslots[day]) for day in assigned_days):
                continue
            
            start_times = [timeslots[day][slot_index][0] for day in assigned_days]
            end_times = [timeslots[day][slot_index + required_slots - 1][1] for day in assigned_days]
            
            if len(set(start_times)) == 1 and len(set(end_times)) == 1:
                available_slots.append((start_times[0], end_times[0], slot_index))

        # Prioritize unused timeslots
        available_slots.sort(key=lambda x: min(timeslot_usage[room['classroom']][day].get(x[2], float('inf')) 
                                               for room in regular_classrooms + lab_classrooms 
                                               for day in assigned_days))

        # Choose classroom
        enr_cap = int(course['Enr Cap']) if pd.notna(course['Enr Cap']) else 30  # Default to 30 if NaN
        if is_lab:
            suitable_classrooms = [room for room in lab_classrooms if room['capacity'] >= enr_cap]
            if not suitable_classrooms:
                suitable_classrooms = [max(lab_classrooms, key=lambda x: x['capacity'])]

            suitable_classrooms.sort(key=lambda x: (lab_classroom_usage[x['classroom']], -x['capacity']))

            # Try to find an available time slot in the least used lab classroom
            assigned_classroom = None
            assigned_time = None
            assigned_slot_index = None
            for classroom in suitable_classrooms:
                for start_time, end_time, slot_index in available_slots:
                    if len(assigned_days) == 1:
                        # Single day assignment
                        if all(not (start_time < existing_end and end_time > existing_start)
                               for existing_start, existing_end in classroom_schedule[classroom['classroom']][assigned_days]):
                            assigned_classroom = classroom['classroom']
                            assigned_time = (start_time, end_time)
                            assigned_slot_index = slot_index
                            break
                    else:
                        # Multi-day assignment
                        if all(not (start_time < existing_end and end_time > existing_start)
                               for day in assigned_days
                               for existing_start, existing_end in classroom_schedule[classroom['classroom']][day]):
                            assigned_classroom = classroom['classroom']
                            assigned_time = (start_time, end_time)
                            assigned_slot_index = slot_index
                            break
                if assigned_classroom:
                    break

            if not assigned_classroom or not assigned_time:
                # If no slot is available, assign to the least used lab classroom
                assigned_classroom = suitable_classrooms[0]['classroom']
                assigned_time, assigned_slot_index = random.choice(available_slots)[:2], random.choice(available_slots)[2]

            # Update lab classroom usage
            lab_classroom_usage[assigned_classroom] += 1
        else:
            suitable_classrooms = [room for room in regular_classrooms if room['capacity'] >= enr_cap]
            if not suitable_classrooms:
                suitable_classrooms = regular_classrooms  # Use all regular classrooms if none meet capacity

            # Shuffle the list of suitable classrooms to promote variety
            random.shuffle(suitable_classrooms)

            # Find available classroom and time slot
            assigned_classroom = None
            assigned_time = None
            assigned_slot_index = None
            for classroom in suitable_classrooms:
                for start_time, end_time, slot_index in available_slots:
                    if len(assigned_days) == 1:
                        # Single day assignment
                        if all(not (start_time < existing_end and end_time > existing_start)
                               for existing_start, existing_end in classroom_schedule[classroom['classroom']][assigned_days]):
                            assigned_classroom = classroom['classroom']
                            assigned_time = (start_time, end_time)
                            assigned_slot_index = slot_index
                            break
                    else:
                        # Multi-day assignment
                        if all(not (start_time < existing_end and end_time > existing_start)
                               for day in assigned_days
                               for existing_start, existing_end in classroom_schedule[classroom['classroom']][day]):
                            assigned_classroom = classroom['classroom']
                            assigned_time = (start_time, end_time)
                            assigned_slot_index = slot_index
                            break
                if assigned_classroom:
                    break

            if not assigned_classroom or not assigned_time:
                # If no slot is available, assign randomly (will be penalized in fitness function)
                assigned_classroom = random.choice(suitable_classrooms)['classroom']
                assigned_time, assigned_slot_index = random.choice(available_slots)[:2], random.choice(available_slots)[2]

        # Update schedules
        for day in assigned_days:
            classroom_schedule[assigned_classroom][day].append((assigned_time[0], assigned_time[1]))
            if assigned_slot_index in timeslot_usage[assigned_classroom][day]:
                timeslot_usage[assigned_classroom][day][assigned_slot_index] += 1
            else:
                timeslot_usage[assigned_classroom][day][assigned_slot_index] = 1
        
        if course['Instructor'] != 'TBD':
            if course['Instructor'] not in instructor_schedule:
                instructor_schedule[course['Instructor']] = {day: [] for day in 'MTWRF'}
            for day in assigned_days:
                instructor_schedule[course['Instructor']][day].append((assigned_time[0], assigned_time[1]))

        # Add the course to the schedule
        schedule.append({
            'Class #': course['Class #'],
            'Subject': course['Subject'],
            'Cat#': course['Cat#'],
            'Sect#': course['Sect#'],
            'Course Name': course['Course Name'],
            'Hrs': hrs,
            'Instructor': course['Instructor'],
            'Enr Cap': enr_cap,
            'Days': assigned_days,
            'Time Start': assigned_time[0],
            'Time End': assigned_time[1],
            'Room #': assigned_classroom,
            'Room Capacity': next(room['capacity'] for room in suitable_classrooms if room['classroom'] == assigned_classroom)
        })

    return schedule, classroom_schedule, instructor_schedule



# Modified fitness function
def fitness_func(ga_instance, solution, solution_idx, courses_cleaned, possible_days, timeslots, possible_lab_days, regular_classrooms, lab_classrooms):
    schedule, classroom_schedule, instructor_schedule = decode_chromosome(solution, courses_cleaned, possible_days, timeslots, possible_lab_days, regular_classrooms, lab_classrooms)

    fitness = 0
    day_distribution = {'M': 0, 'T': 0, 'W': 0, 'R': 0, 'F': 0, 'MWF': 0, 'TR': 0}
    classroom_usage = {room['classroom']: 0 for room in regular_classrooms + lab_classrooms}
    total_timeslots = sum(len(slots) for slots in timeslots.values())
    
    lab_classroom_conflicts = 0
    classroom_conflicts = 0
    instructor_conflicts = 0
    invalid_day_assignments = 0
    invalid_time_assignments = 0
    incorrect_lab_assignments = 0
    insufficient_capacity = 0
    incorrect_duration = 0
    unassigned_rooms = 0
    lab_room_penalty = 0
    regular_room_penalty = 0

    lab_classroom_usage = {room['classroom']: 0 for room in lab_classrooms}

    for course in schedule:
        if 'IsHeader' in course and course['IsHeader']:
            continue

        days = course.get('Days', '')
        if not isinstance(days, str):
            days = str(days) if pd.notna(days) else ''

        # Skip the room usage calculation if the room is unavailable
        room_number = course['Room #']
        if room_number == 'unavailable':
            continue

        # 1. Conflict-free scheduling
        for day in days:
            if day not in 'MTWRF':
                invalid_day_assignments += 1
                continue
            
            # Check classroom conflicts
            overlaps = sum(1 for start, end in classroom_schedule.get(course['Room #'], {}).get(day, [])
                           if start < course['Time End'] and end > course['Time Start'])
            if overlaps > 1:
                classroom_conflicts += 1

            # Check instructor conflicts
            if course['Instructor'] != 'TBD':
                overlaps = sum(1 for start, end in instructor_schedule.get(course['Instructor'], {}).get(day, [])
                               if start < course['Time End'] and end > course['Time Start'])
                if overlaps > 1:
                    instructor_conflicts += 1

        # 2. Valid day and time assignments
        if days not in ['M', 'T', 'W', 'R', 'F', 'MWF', 'TR']:
            invalid_day_assignments += 1
        if not course['Time Start'] or not course['Time End']:
            invalid_time_assignments += 1
        else:
            fitness += 1  # Reward for valid time assignment

        # 3. Balance between MWF and TR courses
        if days in day_distribution:
            day_distribution[days] += 1

        # 4. Proper assignment of lab sections
        if 'L' in str(course['Sect#']):
            lecture_days = schedule[schedule.index(course) - 1]['Days']
            if days not in possible_lab_days.get(lecture_days, []):
                incorrect_lab_assignments += 1

        # 5. Appropriate classroom capacity
        room_capacity = next((classroom['capacity'] for classroom in regular_classrooms + lab_classrooms if classroom['classroom'] == room_number), 0)
        if course['Enr Cap'] > room_capacity:
            insufficient_capacity += 1

        # 6. Correct duration of classes
        hrs = course['Hrs']
        if pd.notna(hrs):
            required_slots = 2 if hrs > 3 else 1
            assigned_slots = sum(1 for slot in timeslots[days[0]] if course['Time Start'] <= slot[0] < course['Time End'])
            if assigned_slots != required_slots:
                incorrect_duration += 1

        # 7. Efficient use of classrooms
        if pd.notna(course['Room #']) and course['Room #'] in classroom_usage:
            classroom_usage[course['Room #']] += len(days)
        else:
            unassigned_rooms += 1
            fitness -= 10  # Penalty for unassigned room

        # 8. Penalize lab courses assigned to regular rooms and vice versa
        is_lab = 'L' in str(course['Sect#'])
        if is_lab and room_number not in [room['classroom'] for room in lab_classrooms]:
            lab_room_penalty += 1
        elif not is_lab and room_number in [room['classroom'] for room in lab_classrooms]:
            regular_room_penalty += 1

        # Check for lab classroom conflicts
        if 'L' in str(course['Sect#']):
            overlaps = sum(1 for start, end in classroom_schedule.get(room_number, {}).get(day, [])
                           if start < course['Time End'] and end > course['Time Start'])
            if overlaps > 1:
                lab_classroom_conflicts += 1

            lab_classroom_usage[room_number] += 1

        # 9. CIT Conc and Adv Opt conflict constraints with ISTE 500 and ISTE 501
        if course.get('Yr Level/ Reqrmt', '') in ['CIT Conc', 'Adv Opt']:  # Handle missing key
            for other_course in schedule:
                if other_course['Class #'] == course['Class #']:
                    continue
                if other_course['Course Name'] in ['ISTE 500', 'ISTE 501']:
                    if days == other_course['Days'] and not (course['Time End'] <= other_course['Time Start'] or course['Time Start'] >= other_course['Time End']):
                        fitness -= 1000  # High penalty for conflict

    # Calculate fitness components
    conflict_penalty = (classroom_conflicts + instructor_conflicts) * 1000
    invalid_assignment_penalty = (invalid_day_assignments + invalid_time_assignments) * 100
    lab_assignment_penalty = incorrect_lab_assignments * 50
    capacity_penalty = insufficient_capacity * 50
    duration_penalty = incorrect_duration * 50

    # Balance between MWF and TR courses
    mwf_count = day_distribution['MWF'] + day_distribution['M'] + day_distribution['W'] + day_distribution['F']
    tr_count = day_distribution['TR'] + day_distribution['T'] + day_distribution['R']
    total_courses = mwf_count + tr_count
    if total_courses > 0:
        balance_ratio = min(mwf_count, tr_count) / total_courses
        balance_score = balance_ratio * 100
    else:
        balance_score = 0

    # Efficient use of classrooms
    total_usage = sum(classroom_usage.values())
    max_possible_usage = len(classroom_usage) * total_timeslots
    efficiency_score = (total_usage / max_possible_usage) * 100

    # Penalize uneven usage of lab classrooms
    lab_usage_values = list(lab_classroom_usage.values())
    lab_usage_range = max(lab_usage_values) - min(lab_usage_values)
    lab_usage_penalty = lab_usage_range * 100  # Increased penalty for uneven usage

    # Penalize lab classroom conflicts more heavily
    lab_conflict_penalty = lab_classroom_conflicts * 2000

    # Calculate final fitness
    fitness += balance_score + efficiency_score
    fitness -= (conflict_penalty + invalid_assignment_penalty + lab_assignment_penalty + 
                capacity_penalty + duration_penalty + lab_room_penalty + regular_room_penalty + lab_usage_penalty + lab_conflict_penalty)

    # Add penalty for unassigned rooms to the final fitness calculation
    fitness -= unassigned_rooms * 50

    return fitness

def resolve_conflicts(schedule):
    unavailable_classroom = 'unavailable'
    classroom_schedule = {}

    # First, build an accurate classroom schedule from the final schedule
    for course in schedule:
        if 'IsHeader' in course and course['IsHeader']:
            continue
        room = course['Room #']
        days = course['Days']
        start = course['Time Start']
        end = course['Time End']
        
        # Skip courses with invalid or missing data
        if pd.isna(room) or pd.isna(days) or pd.isna(start) or pd.isna(end):
            continue
        
        # Ensure days is a string
        days = str(days) if pd.notna(days) else ''
        
        if room not in classroom_schedule:
            classroom_schedule[room] = {}
        for day in days:
            if day not in classroom_schedule[room]:
                classroom_schedule[room][day] = []
            classroom_schedule[room][day].append((start, end, course))

    # Now resolve conflicts
    for room, days in classroom_schedule.items():
        for day, timeslots in days.items():
            # Sort timeslots by start time
            timeslots.sort(key=lambda x: x[0])
            
            # Check for overlaps
            for i in range(len(timeslots)):
                for j in range(i+1, len(timeslots)):
                    if timeslots[i][1] > timeslots[j][0]:  # Overlap found
                        # Move the later course to 'unavailable'
                        timeslots[j][2]['Room #'] = unavailable_classroom

    return schedule

def on_generation(ga_instance):
    print(f"Generation {ga_instance.generations_completed}")

# Modified generate_schedule function
def generate_schedule(courses_cleaned, semester, num_generations = 20):
    regular_classrooms, lab_classrooms = load_classrooms('excel/classrooms.csv')
    fixed_courses, non_fixed_courses = separate_courses(courses_cleaned)

    possible_days = ['MWF', 'TR']
    possible_lab_days = {'MWF': ['M', 'W', 'F'], 'TR': ['T', 'R']}
    num_courses = len(non_fixed_courses)
    
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
                           num_parents_mating=20,
                           fitness_func=fitness_wrapper,
                           num_genes=num_courses * 3,
                           gene_space=gene_space,
                           sol_per_pop=100,  # Increased population size
                           parent_selection_type="tournament",
                           K_tournament=5,
                           keep_parents=2,
                           crossover_type="two_points",
                           mutation_type="random",
                           mutation_percent_genes=10,
                           on_generation=on_generation)
    
    ga_instance.run()
    
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_schedule, _, _ = decode_chromosome(solution, non_fixed_courses, possible_days, timeslots, possible_lab_days, regular_classrooms, lab_classrooms)
    
    # Combine fixed and non-fixed courses before resolving conflicts
    combined_schedule = fixed_courses.to_dict('records') + best_schedule
    
    # Resolve conflicts after the last generation
    resolved_schedule = resolve_conflicts(combined_schedule)
    
    best_schedule_df = pd.DataFrame(resolved_schedule)
    
    columns_order = ['Class #', 'Subject', 'Cat#', 'Sect#', 'Course Name', 'Hrs', 'Instructor', 'Enr Cap', 'Days', 'Time Start', 'Time End', 'Room #']
    best_schedule_df = best_schedule_df.reindex(columns=columns_order)
    
    output_path = f'excel/Best_Schedule_{semester.capitalize()}.csv'
    best_schedule_df.to_csv(output_path, index=False)
    
    return output_path

# Modify the last part of the script to use multithreading
if __name__ == "__main__":
    # Load and preprocess the spring and fall course data
    spring_courses_df, spring_courses = load_and_preprocess('excel/Spring_2024_Filtered_Corrected_Updated_v4.csv')
    fall_courses_df, fall_courses = load_and_preprocess('excel/Fall_2023_Filtered_Corrected_Updated_v4.csv')

    # Use ThreadPoolExecutor to run the schedule generation concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        spring_future = executor.submit(generate_schedule, spring_courses, 'spring')
        fall_future = executor.submit(generate_schedule, fall_courses, 'fall')

        # Wait for both tasks to complete and get the results
        spring_schedule_path = spring_future.result()
        fall_schedule_path = fall_future.result()

    # Print the paths to the generated schedules
    print("Spring schedule path:", spring_schedule_path)
    print("Fall schedule path:", fall_schedule_path)