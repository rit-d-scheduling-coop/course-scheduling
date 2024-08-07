import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def visualize_instructor_schedule(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert time strings to datetime objects
    def parse_time(time_str):
        try:
            return pd.to_datetime(time_str, format='%H:%M').time()
        except ValueError:
            try:
                return pd.to_datetime(time_str, format='%H:%M:%S').time()
            except ValueError:
                try:
                    return pd.to_datetime(time_str, format='%I:%M %p').time()
                except ValueError:
                    return pd.to_datetime('19:00', format='%H:%M').time()

    df['Time Start'] = df['Time Start'].apply(parse_time)
    df['Time End'] = df['Time End'].apply(parse_time)
    
    # Create a unique identifier for each class
    df['Class'] = df['Subject'].astype(str) + ' ' + df['Cat#'].astype(str) + ' ' + df['Sect#'].astype(str)
    
    # Handle mixed data types in 'Room #' column
    df['Room #'] = df['Room #'].astype(str)
    
    # Create a color palette
    color_palette = sns.color_palette("husl", n_colors=len(df['Class'].unique()))
    color_dict = dict(zip(df['Class'].unique(), color_palette))
    
    # Function to plot for a specific instructor
    def plot_instructor_schedule(instructor_df, ax, instructor_name):
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_map = {'M': 0, 'T': 1, 'W': 2, 'R': 3, 'F': 4}
        
        # Create a dictionary to store conflicting classes
        conflicts = {day: {} for day in range(5)}
        
        # Identify conflicts and combine them
        for _, row in instructor_df.iterrows():
            if pd.notna(row['Time Start']) and pd.notna(row['Time End']):
                start = row['Time Start']
                end = row['Time End']
                duration = ((end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)) / 60
                start_time = start.hour + start.minute/60
                
                for day in str(row['Days']):
                    day_index = day_map.get(day)
                    if day_index is None:
                        continue
                    
                    key = (start_time, duration)
                    if key in conflicts[day_index]:
                        conflicts[day_index][key].append(f"{row['Class']} - {row['Room #']}")
                    else:
                        conflicts[day_index][key] = [f"{row['Class']} - {row['Room #']}"]
        
        # Plot combined classes
        for day_index, day_conflicts in conflicts.items():
            for (start_time, duration), classes in day_conflicts.items():
                combined_class = ' / '.join(classes)
                color = color_dict[classes[0].split(' - ')[0]]  # Use color of first class
                
                rect = ax.barh(day_index, duration, left=start_time, height=0.5,
                               align='center', color=color, alpha=0.8)
                
                # Add text label inside the rectangle
                rx, ry = rect[0].get_xy()
                cx = rx + rect[0].get_width()/2.0
                cy = ry + rect[0].get_height()/2.0
                ax.text(cx, cy, combined_class, ha='center', va='center',
                        rotation=0, fontsize=8, color='black', fontweight='bold')
        
        # Customize the plot
        ax.set_ylim(-1, len(days))
        ax.set_xlim(7, 19)  # Assuming classes are between 7 AM and 7 PM
        ax.set_yticks(range(len(days)))
        ax.set_yticklabels(days)
        ax.set_xlabel('Time')
        ax.set_ylabel('Day')
        ax.set_title(f"{instructor_name}'s Weekly Schedule")
        
        # Add x-axis labels for each hour
        ax.set_xticks(range(7, 20))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(7, 20)])
    
    # Get unique instructors
    instructors = df['Instructor'].unique()
    
    # Create a separate figure for each instructor
    for instructor in instructors:
        instructor_df = df[df['Instructor'] == instructor]
        
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_instructor_schedule(instructor_df, ax, instructor)
        
        plt.tight_layout()
        plt.show()

# Call the function with the path to your CSV files
visualize_instructor_schedule('./Excel/Best_Schedule_Fall.csv')
visualize_instructor_schedule('./Excel/Best_Schedule_Spring.csv')