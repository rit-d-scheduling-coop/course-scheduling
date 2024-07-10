import pandas as pd

# Load the Excel file
file_path = '2023-2024 Plan for Dept Heads.xlsx'
xls = pd.ExcelFile(file_path)

# Load the 2nd tab with header at the correct row
df_spring_2023 = pd.read_excel(xls, sheet_name='Spring 2023', header=2)

# Load the 3rd tab, skip the first two rows, and then set the correct header row
df_fall_2022 = pd.read_excel(xls, sheet_name='Fall 2022', skiprows=2)
df_fall_2022.columns = df_fall_2022.iloc[0]  # Set the correct header
df_fall_2022 = df_fall_2022[1:]  # Skip the header row

# Drop the specified columns
columns_to_drop = ['Estimated NEED', 'Estimated ELIGIBLE', 'Day', 'Start Time', 'End Time', 
                   'Zoom Links (for Hybrid courses and students granted exceptions to attend online ONLY)']
df_spring_2023 = df_spring_2023.drop(columns=columns_to_drop, errors='ignore')
df_fall_2022 = df_fall_2022.drop(columns=columns_to_drop, errors='ignore')

# Drop the first instances of "Day", "Start Time", and "End Time" but keep the second instances
df_spring_2023 = df_spring_2023.loc[:, ~df_spring_2023.columns.duplicated()]
df_fall_2022 = df_fall_2022.loc[:, ~df_fall_2022.columns.duplicated()]

# Remove leading and trailing white spaces from all columns
df_spring_2023 = df_spring_2023.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df_fall_2022 = df_fall_2022.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# List of wanted departments
wanted_departments = [
'COMPUTING', 'BS CIT', 'BS COMPUTING SECURITY', 'MATH/SCIENCE', 'ASC'
]

# Adjusted function to filter the rows based on department names and their courses
def filter_departments_corrected(df, departments, class_col, subject_col, cat_col, sect_col, course_name_col):
    # Initialize an empty DataFrame to store the filtered rows
    filtered_df = pd.DataFrame(columns=df.columns)
    add_row = False

    for _, row in df.iterrows():
        # Check if the row is a department header
        if pd.isna(row[class_col]) and pd.isna(row[subject_col]) and pd.isna(row[cat_col]) and pd.isna(row[sect_col]):
            if row[course_name_col] in departments:
                add_row = True
            else:
                add_row = False
        # Add rows that belong to the wanted departments
        if add_row:
            filtered_df = pd.concat([filtered_df, pd.DataFrame([row], columns=df.columns)], ignore_index=True)

    return filtered_df

# Apply the function to both dataframes with the corrected column names
df_spring_2023_filtered_corrected = filter_departments_corrected(df_spring_2023, wanted_departments, 'Class #', 'Sbjct', 'Crs #', 'Sect#', 'Course Name')
df_fall_2022_filtered_corrected = filter_departments_corrected(df_fall_2022, wanted_departments, 'Class #', 'Subject', 'Cat#', 'Sect#', 'Course Name')

# Save the filtered dataframes to new CSV files
csv_spring_2023_filtered_corrected = 'Spring_2023_Filtered_Corrected.csv'
csv_fall_2022_filtered_corrected = 'Fall_2022_Filtered_Corrected.csv'

df_spring_2023_filtered_corrected.to_csv(csv_spring_2023_filtered_corrected, index=False)
df_fall_2022_filtered_corrected.to_csv(csv_fall_2022_filtered_corrected, index=False)

(csv_spring_2023_filtered_corrected, csv_fall_2022_filtered_corrected)