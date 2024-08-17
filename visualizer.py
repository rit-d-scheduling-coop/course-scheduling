import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def visualize_schedule(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert time strings to datetime objects
    def parse_time(time_str):
        try:
            return pd.to_datetime(time_str, format='%H:%M:%S').time()
        except ValueError:
            try:
                return pd.to_datetime(time_str, format='%H:%M').time()
            except ValueError:
                return pd.to_datetime('19:00', format='%H:%M').time()

    df['Time Start'] = df['Time Start'].apply(parse_time)
    df['Time End'] = df['Time End'].apply(parse_time)
    
    # Create a unique identifier for each class
    df['Class'] = df['Subject'].astype(str) + ' ' + df['Cat#'].astype(str) + ' ' + df['Sect#'].astype(str)
    
    # Handle mixed data types in 'Room #' column
    df['Room #'] = df['Room #'].astype(str)
    
    # Create a list of all unique rooms, excluding NaN and empty string values
    rooms = sorted(df['Room #'].dropna().replace('', pd.NA).dropna().unique())
    
    # Create a color palette
    color_palette = sns.color_palette("husl", n_colors=len(df['Class'].unique()))
    color_dict = dict(zip(df['Class'].unique(), color_palette))
    
    # Function to plot for a specific day
    def plot_day(day_df, ax, title):
        # Create a dictionary to store overlapping classes
        overlaps = {}
        
        # Plot each class as a rectangle with labels inside
        for _, row in day_df.iterrows():
            if pd.notna(row['Room #']) and row['Room #'] != '' and pd.notna(row['Time Start']) and pd.notna(row['Time End']):
                start = datetime.combine(datetime.today(), row['Time Start'])
                end = datetime.combine(datetime.today(), row['Time End'])
                duration = (end - start).total_seconds() / 3600  # duration in hours
                
                # Check for overlaps
                key = (row['Room #'], start.hour + start.minute/60, end.hour + end.minute/60)
                if key in overlaps:
                    overlaps[key].append(row['Class'])
                else:
                    overlaps[key] = [row['Class']]
        
        # Plot the classes, combining overlaps
        for (room, start, end), classes in overlaps.items():
            duration = end - start
            y_pos = rooms.index(room)
            rect = ax.barh(y_pos, duration, left=start, height=0.5, align='center', 
                    color=color_dict[classes[0]], alpha=0.8)
            
            # Add text label inside the rectangle
            rx, ry = rect[0].get_xy()
            cx = rx + rect[0].get_width()/2.0
            cy = ry + rect[0].get_height()/2.0
            label = ' / '.join(str(c) for c in classes)  # Convert each class to string
            ax.text(cx, cy, label, ha='center', va='center', rotation=0, 
                    fontsize=8, color='black', fontweight='bold')
        
        # Customize the plot
        ax.set_ylim(-1, len(rooms))
        ax.set_xlim(7, 19)  # Assuming classes are between 7 AM and 7 PM
        ax.set_yticks(range(len(rooms)))
        ax.set_yticklabels(rooms)
        ax.set_xlabel('Time')
        ax.set_ylabel('Room')
        ax.set_title(title)
        
        # Add x-axis labels for each hour
        ax.set_xticks(range(7, 20))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(7, 20)])
    
    # Create five subplots, one for each day
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(20, 50))
    
    # Separate data for each day
    days = ['M', 'T', 'W', 'R', 'F']
    day_dfs = []
    for day in days:
        day_df = df[df['Days'].str.contains(day, na=False)]
        day_dfs.append(day_df)
    
    # Plot schedule for each day
    for i, (day, day_df, ax) in enumerate(zip(days, day_dfs, [ax1, ax2, ax3, ax4, ax5])):
        plot_day(day_df, ax, f'{file_path} - {day} Schedule')
    
    plt.tight_layout()
    plt.show()

# Call the function with the path to your CSV files
# visualize_schedule('./Excel/Best_Schedule_Fall.csv')
# visualize_schedule('./Excel/Best_Schedule_Spring.csv')