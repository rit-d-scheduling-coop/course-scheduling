import pandas as pd

# Load the Excel file
file_path = '2023-2024 Plan for Dept Heads.xlsx'
xls = pd.ExcelFile(file_path)

# Load the Fall 2023 tab with header at the correct row
df_fall_2023 = pd.read_excel(xls, sheet_name='Fall 2023 (2231)', header=2)

# Load the Spring 2024 tab with header at the correct row
df_spring_2024 = pd.read_excel(xls, sheet_name='Spring 2024', header=2)

# Drop columns starting with 'Unnamed:'
df_fall_2023 = df_fall_2023.loc[:, ~df_fall_2023.columns.str.startswith('Unnamed:')]
df_spring_2024 = df_spring_2024.loc[:, ~df_spring_2024.columns.str.startswith('Unnamed:')]

# Rename the second instances of Days, Time Start, and Time End to remove .1
df_fall_2023 = df_fall_2023.rename(columns={
    'Days.1': 'Days',
    'Time Start.1': 'Time Start',
    'Time End.1': 'Time End'
})
df_spring_2024 = df_spring_2024.rename(columns={
    'Days.1': 'Days',
    'Time Start.1': 'Time Start',
    'Time End.1': 'Time End'
})

# Drop any remaining duplicate columns
df_fall_2023 = df_fall_2023.loc[:, ~df_fall_2023.columns.duplicated()]
df_spring_2024 = df_spring_2024.loc[:, ~df_spring_2024.columns.duplicated()]

# Drop the specified columns
columns_to_drop = ['Yr Level/ Reqrmt','Course Attribute','Crdt','Estimated NEED', 'Estimated ELIGIBLE', 'Zoom Links (for Hybrid courses and students granted exceptions to attend online ONLY)']
df_fall_2023 = df_fall_2023.drop(columns=columns_to_drop, errors='ignore')
df_spring_2024 = df_spring_2024.drop(columns=columns_to_drop, errors='ignore')

# Remove leading and trailing white spaces from all string columns
df_fall_2023 = df_fall_2023.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df_spring_2024 = df_spring_2024.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# List of wanted departments
wanted_departments = [
    'COMPUTING', 'BS CIT', 'BS COMPUTING SECURITY', 'MATH/SCIENCE',
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
df_fall_2023_filtered_corrected = filter_departments_corrected(df_fall_2023, wanted_departments, 'Class #', 'Subject', 'Cat#', 'Sect#', 'Course Name')
df_spring_2024_filtered_corrected = filter_departments_corrected(df_spring_2024, wanted_departments, 'Class #', 'Subject', 'Cat#', 'Sect#', 'Course Name')

# Save the filtered dataframes to new CSV files
csv_fall_2023_filtered_corrected = 'Fall_2023_Filtered_Corrected.csv'
csv_spring_2024_filtered_corrected = 'Spring_2024_Filtered_Corrected.csv'

df_fall_2023_filtered_corrected.to_csv(csv_fall_2023_filtered_corrected, index=False)
df_spring_2024_filtered_corrected.to_csv(csv_spring_2024_filtered_corrected, index=False)

(csv_fall_2023_filtered_corrected, csv_spring_2024_filtered_corrected)
