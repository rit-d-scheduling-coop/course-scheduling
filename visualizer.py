import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def visualize_schedule(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert time strings to datetime objects
    df['Time Start'] = pd.to_datetime(df['Time Start'], format='%H:%M').dt.time
    df['Time End'] = pd.to_datetime(df['Time End'], format='%H:%M').dt.time
    
    # Create a unique identifier for each class
    df['Class'] = df['Subject'] + ' ' + df['Cat#'].astype(str) + ' ' + df['Sect#'].astype(str)
    
    # Handle mixed data types in 'Room #' column
    df['Room #'] = df['Room #'].astype(str)
    
    # Create a list of all unique rooms, excluding NaN values
    rooms = sorted(df['Room #'].dropna().unique())
       
    # Create a color palette
    color_palette = sns.color_palette("husl", n_colors=len(df['Class'].unique()))
    color_dict = dict(zip(df['Class'].unique(), color_palette))
    
    # Separate data for MWF and TR
    df_mwf = df[df['Days'].isin(['M', 'W', 'F', 'MWF'])]
    df_tr = df[df['Days'].isin(['T', 'R', 'TR'])]
    
    # Function to plot for a specific day group
    def plot_day_group(day_df, ax, title):
        # Create a dictionary to store overlapping classes
        overlaps = {}
        
        # Plot each class as a rectangle with labels inside
        for _, row in day_df.iterrows():
            if pd.notna(row['Room #']) and pd.notna(row['Time Start']) and pd.notna(row['Time End']):
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
            rect = ax.barh(room, duration, left=start, height=0.5, align='center', 
                    color=color_dict[classes[0]], alpha=0.8)
            
            # Add text label inside the rectangle
            rx, ry = rect[0].get_xy()
            cx = rx + rect[0].get_width()/2.0
            cy = ry + rect[0].get_height()/2.0
            label = ' / '.join(classes)
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
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))
    
    # Plot MWF schedule
    plot_day_group(df_mwf, ax1, f'{file_path} - MWF Schedule')
    
    # Plot TR schedule
    plot_day_group(df_tr, ax2, f'{file_path} - TR Schedule')
    
    plt.tight_layout()
    plt.show()

# Call the function with the path to your CSV files
visualize_schedule('Best_Schedule_Fall.csv')
visualize_schedule('Best_Schedule_Spring.csv')
